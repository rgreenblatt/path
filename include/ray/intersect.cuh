#pragma once

#include "ray/best_intersection.h"
#include "ray/cone.h"
#include "ray/cube.h"
#include "ray/cylinder.h"
#include "ray/kdtree.h"
#include "ray/ray_utils.h"
#include "ray/sphere.h"
#include "scene/shape.h"
#include "scene/shape_data.h"

namespace ray {
namespace detail {

template <typename T> __global__ void fill(T *data, unsigned size, T value) {
  unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    data[index] = value;
  }
}

inline __host__ __device__ void
initial_world_space_directions(unsigned x, unsigned y, unsigned width,
                               unsigned height,
                               const Eigen::Vector3f &world_space_eye,
                               const scene::Transform &m_film_to_world,
                               Eigen::Vector3f *world_space_directions) {
  unsigned index = x + y * width;

  if (x >= width || y >= height) {
    return;
  }

  const Eigen::Vector3f camera_space_film_plane(
      (2.0f * static_cast<float>(x)) / static_cast<float>(width) - 1.0f,
      (-2.0f * static_cast<float>(y)) / static_cast<float>(height) + 1.0f,
      -1.0f);
  const auto world_space_film_plane = m_film_to_world * camera_space_film_plane;
  world_space_directions[index] =
      (world_space_film_plane - world_space_eye).normalized();
}

__global__ void
initial_world_space_directions(unsigned width, unsigned height,
                               const Eigen::Vector3f world_space_eye,
                               const scene::Transform m_film_to_world,
                               Eigen::Vector3f *world_space_directions) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

  initial_world_space_directions(x, y, width, height, world_space_eye,
                                 m_film_to_world, world_space_directions);
}

inline void
initial_world_space_directions_cpu(unsigned width, unsigned height,
                                   const Eigen::Vector3f &world_space_eye,
                                   const scene::Transform &m_film_to_world,
                                   Eigen::Vector3f *world_space_directions) {
  for (unsigned x = 0; x < width; x++) {
    for (unsigned y = 0; y < height; y++) {
      initial_world_space_directions(x, y, width, height, world_space_eye,
                                     m_film_to_world, world_space_directions);
    }
  }
}

template <bool normal_and_uv>
__host__ __device__ IntersectionOp<normal_and_uv>
solve_type(scene::Shape type, const Eigen::Vector3f &point,
           const Eigen::Vector3f &direction, bool texture_map, unsigned index) {
  switch (type) {
  case scene::Shape::Sphere:
    return solve_sphere<normal_and_uv>(point, direction, texture_map, index);
  case scene::Shape::Cylinder:
    return solve_cylinder<normal_and_uv>(point, direction, texture_map);
  case scene::Shape::Cube:
    return solve_cube<normal_and_uv>(point, direction, texture_map);
  case scene::Shape::Cone:
    return solve_cone<normal_and_uv>(point, direction, texture_map);
  }
}

template <bool normal_and_uv>
__host__
    __device__ inline thrust::optional<BestIntersectionGeneral<normal_and_uv>>
    get_shape_intersection(const scene::ShapeData *shapes,
                           const unsigned shape_idx,
                           const scene::Shape shape_type,
                           const Eigen::Vector3f &world_space_eye,
                           const Eigen::Vector3f &world_space_direction,
                           unsigned index) {
  const auto &shape = shapes[shape_idx];
  const auto object_space_eye = shape.get_world_to_object() * world_space_eye;
  const auto object_space_direction =
      shape.get_world_to_object().linear() * world_space_direction;

  return optional_map(
      solve_type<normal_and_uv>(
          shape_type, object_space_eye, object_space_direction,
          normal_and_uv && shape.get_material().texture_map_index.has_value(),
          index),
      [=](const auto &value) {
        return BestIntersectionGeneral<normal_and_uv>(value, shape_idx);
      });
}

__host__ __device__ inline void solve_intersection(
    unsigned x, unsigned y, unsigned width, unsigned height,
    unsigned start_shape, const scene::ShapeData *shapes,
    const Eigen::Vector3f *world_space_eyes,
    const Eigen::Vector3f *world_space_directions, const KDTreeNode *nodes,
    unsigned root_node_count, const unsigned *ignores, const uint8_t *disables,
    thrust::optional<BestIntersectionNormalUV> *best_intersections,
    const scene::Shape shape_type, bool is_first) {
  unsigned index = x + y * width;

  if (x >= width || y >= height) {
    return;
  }

  auto &best_normals_uv = best_intersections[index];

  if (!is_first && disables[index]) {
    best_normals_uv =
        thrust::optional<BestIntersectionNormalUV>(thrust::nullopt);

    return;
  }

  const auto &world_space_direction = world_space_directions[index];
  const auto &world_space_eye = world_space_eyes[index];

  thrust::optional<BestIntersection> best = thrust::nullopt;

  if (root_node_count != 0) {
    std::array<std::array<unsigned, 2>, 64> node_stack;
    node_stack[0] = {root_node_count - 1, 0};
    uint8_t node_stack_size = 1;

    while (node_stack_size != 0) {
      thrust::optional<std::array<unsigned, 2>> start_end = thrust::nullopt;

      while (!start_end.has_value() && node_stack_size != 0) {
        const auto &stack_v = node_stack[node_stack_size - 1];

        const auto &current_node = nodes[stack_v[0]];
        auto depth = stack_v[1];

        auto bounding_intersection = current_node.solveBoundingIntersection(
            world_space_eye, world_space_direction);
        if (bounding_intersection &&
            (!best || best->intersection > *bounding_intersection)) {
          current_node.case_split_or_data(
              [&](const KDTreeSplit &split) {
                const auto intersection_point =
                    world_space_eye +
                    world_space_direction * *bounding_intersection;
                const int axis = depth % 3;
                auto first = split.left_index;
                auto second = split.right_index;

                if (intersection_point[axis] > split.division_point) {
                  auto temp = first;
                  first = second;
                  second = temp;
                }

                unsigned new_depth = depth + 1;
                node_stack[node_stack_size - 1] = {second, new_depth};
                node_stack_size++;
                node_stack[node_stack_size - 1] = {first, new_depth};
                node_stack_size++; // counter act --;
              },
              [&](const std::array<unsigned, 2> &data) {
                start_end = thrust::make_optional(data);
              });
        }

        node_stack_size--;
      }

      if (start_end.has_value()) {
        auto local_start_shape = (*start_end)[0];
        auto local_end_shape = (*start_end)[1];
        for (unsigned shape_idx = start_shape + local_start_shape;
             shape_idx < start_shape + local_end_shape; shape_idx++) {
          if (!is_first && ignores[index] == shape_idx) {
            continue;
          }

          best =
              optional_min(get_shape_intersection<false>(
                               shapes, shape_idx, shape_type, world_space_eye,
                               world_space_direction, index),
                           best);
        }
      }
    }
  }

  if (best.has_value()) {
    // TODO: why required
    auto out = get_shape_intersection<true>(shapes, best->shape_idx, shape_type,
                                            world_space_eye,
                                            world_space_direction, index);
    best_normals_uv = out;
  } else {
    best_normals_uv =
        thrust::optional<BestIntersectionNormalUV>(thrust::nullopt);
  }
}

__global__ void solve_intersections(
    unsigned width, unsigned height, unsigned start_shape,
    const scene::ShapeData *shapes, const Eigen::Vector3f *world_space_eyes,
    const Eigen::Vector3f *world_space_directions, const KDTreeNode *nodes,
    unsigned root_node_count, const unsigned *ignores, const uint8_t *disables,
    thrust::optional<BestIntersectionNormalUV> *best_intersections,
    const scene::Shape shape_type, bool is_first) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

  solve_intersection(x, y, width, height, start_shape, shapes, world_space_eyes,
                     world_space_directions, nodes, root_node_count, ignores,
                     disables, best_intersections, shape_type, is_first);
}

inline void solve_intersections_cpu(
    unsigned width, unsigned height, unsigned start_shape,
    const scene::ShapeData *shapes, const Eigen::Vector3f *world_space_eyes,
    const Eigen::Vector3f *world_space_directions, const KDTreeNode *nodes,
    unsigned root_node_count, const unsigned *ignores, const uint8_t *disables,
    thrust::optional<BestIntersectionNormalUV> *best_intersections,
    const scene::Shape shape_type, bool is_first) {
  for (unsigned x = 0; x < width; x++) {
    for (unsigned y = 0; y < height; y++) {
      solve_intersection(x, y, width, height, start_shape, shapes,
                         world_space_eyes, world_space_directions, nodes,
                         root_node_count, ignores, disables, best_intersections,
                         shape_type, is_first);
    }
  }
}

template <typename... T>
__host__ __device__ void
minimize_intersections(unsigned size, unsigned index,
                       thrust::optional<BestIntersectionNormalUV> *first,
                       T... rest) {
  if (index > size) {
    return;
  }

  // needed????
  auto new_first = optional_min(first[index], rest[index]...);
  first[index] = new_first;
}

template <typename... T>
__global__ void minimize_all_intersections(unsigned size, T... rest) {
  unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  minimize_intersections(size, index, rest...);
}

template <typename... T>
void minimize_all_intersections_cpu(unsigned size, T... rest) {
  for (unsigned index = 0; index < size; index++) {
    minimize_intersections(size, index, rest...);
  }
}
} // namespace detail
} // namespace ray
