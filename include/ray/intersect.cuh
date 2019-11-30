#pragma once

#include "ray/best_intersection.h"
#include "ray/by_type_data.h"
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
  if (x >= width || y >= height) {
    return;
  }

  unsigned index = x + y * width;

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

template <bool normal_and_uv, scene::Shape shape_type>
__host__ __device__ IntersectionOp<normal_and_uv>
solve_type(const Eigen::Vector3f &point, const Eigen::Vector3f &direction,
           bool texture_map) {
  if constexpr (shape_type == scene::Shape::Sphere) {
    return solve_sphere<normal_and_uv>(point, direction, texture_map);
  } else if constexpr (shape_type == scene::Shape::Cylinder) {
    return solve_cylinder<normal_and_uv>(point, direction, texture_map);
  } else if constexpr (shape_type == scene::Shape::Cube) {
    return solve_cube<normal_and_uv>(point, direction, texture_map);
  } else if constexpr (shape_type == scene::Shape::Cone) {
    return solve_cone<normal_and_uv>(point, direction, texture_map);
  }
}

template <bool normal_and_uv, scene::Shape shape_type>
__host__
    __device__ inline thrust::optional<BestIntersectionGeneral<normal_and_uv>>
    get_shape_intersection(const scene::ShapeData *shapes,
                           const unsigned shape_idx,
                           const Eigen::Vector3f &world_space_eye,
                           const Eigen::Vector3f &world_space_direction) {
  const auto &shape = shapes[shape_idx];
  const auto object_space_eye = shape.get_world_to_object() * world_space_eye;
  const auto object_space_direction =
      shape.get_world_to_object().linear() * world_space_direction;

  return optional_map(
      solve_type<normal_and_uv, shape_type>(
          object_space_eye, object_space_direction,
          normal_and_uv && shape.get_material().texture_map_index.has_value()),
      [&](const auto &value) {
        return BestIntersectionGeneral<normal_and_uv>(value, shape_idx);
      });
}

// Rename...
template <scene::Shape shape_type, typename F>
__host__ __device__ inline void solve_general_intersection(
    const ByTypeDataRef &by_type_data,
    const scene::ShapeData *shapes, const Eigen::Vector3f &world_space_eye,
    const Eigen::Vector3f &world_space_direction, const unsigned &ignore_v,
    const uint8_t &disable, thrust::optional<BestIntersection> &best,
    bool is_first, bool use_kd_tree, const F &f) {
  if (!is_first && disable) {
    return;
  }

  auto solve_index = [&](unsigned shape_idx) {
    if (!is_first && ignore_v == shape_idx) {
      return false;
    }

    return f(get_shape_intersection<false, shape_type>(
        shapes, shape_idx, world_space_eye, world_space_direction));
  };

  if (use_kd_tree) {
    auto inv_direction = (1.0f / world_space_direction.array()).eval();

    if (by_type_data.root_node_count != 0) {
      std::array<std::array<unsigned, 2>, 64> node_stack;
      node_stack[0] = {by_type_data.root_node_count - 1, 0};
      uint8_t node_stack_size = 1;

      thrust::optional<std::array<unsigned, 2>> start_end = thrust::nullopt;
      unsigned current_shape_index = 0;

      while (node_stack_size != 0 || start_end.has_value()) {
        while (!start_end.has_value() && node_stack_size != 0) {
          const auto &stack_v = node_stack[node_stack_size - 1];

          const auto &current_node = by_type_data.nodes[stack_v[0]];
          auto depth = stack_v[1];

          auto bounding_intersection = current_node.solveBoundingIntersection(
              world_space_eye, inv_direction);

          if (bounding_intersection &&
              (!best || best->intersection > *bounding_intersection)) {
            current_node.case_split_or_data(
                [&](const KDTreeSplit &split) {
                  const auto intersection_point =
                      world_space_eye +
                      world_space_direction * *bounding_intersection;
                  const unsigned axis = depth % 3;
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
                  current_shape_index = data[0];
                });
          }

          node_stack_size--;
        }

        if (start_end.has_value()) {
          auto local_end_shape = (*start_end)[1];
          unsigned shape_idx = current_shape_index + by_type_data.start_shape;
          current_shape_index++;
          if (current_shape_index >= local_end_shape) {
            start_end = thrust::nullopt;
          }

          if (solve_index(shape_idx)) {
            return;
          }
        }
      }
    }
  } else {
    for (unsigned shape_idx = by_type_data.start_shape;
         shape_idx < by_type_data.start_shape + by_type_data.num_shape;
         shape_idx++) {
      if (solve_index(shape_idx)) {
        return;
      }
    }
  }
}

template <scene::Shape shape_type>
__host__ __device__ inline void
solve_intersection(unsigned x, unsigned y, unsigned width, unsigned height,
                   ByTypeDataRef &by_type_data, const scene::ShapeData *shapes,
                   const Eigen::Vector3f *world_space_eyes,
                   const Eigen::Vector3f *world_space_directions,
                   const unsigned *ignores, const uint8_t *disables,
                   bool is_first, bool use_kd_tree) {
  if (x >= width || y >= height) {
    return;
  }

  unsigned index = x + y * width;

  thrust::optional<BestIntersection> best = thrust::nullopt;

  const auto &world_space_direction = world_space_directions[index];
  const auto &world_space_eye = world_space_eyes[index];

  solve_general_intersection<shape_type>(
      by_type_data, shapes, world_space_eye, world_space_direction,
      ignores[index], disables[index], best, is_first, use_kd_tree,
      [&](const thrust::optional<BestIntersection> &new_best) {
        best = optional_min(best, new_best);

        return false;
      });

  auto &best_normals_uv = by_type_data.intersections[index];

  if (best.has_value()) {
    // TODO: why required
    auto out = get_shape_intersection<true, shape_type>(
        shapes, best->shape_idx, world_space_eye, world_space_direction);
    best_normals_uv = out;
  } else {
    best_normals_uv =
        thrust::optional<BestIntersectionNormalUV>(thrust::nullopt);
  }
}

template <scene::Shape shape_type>
__global__ void solve_intersections(
    unsigned width, unsigned height, ByTypeDataRef by_type_data,
    const scene::ShapeData *shapes, const Eigen::Vector3f *world_space_eyes,
    const Eigen::Vector3f *world_space_directions, const unsigned *ignores,
    const uint8_t *disables, bool is_first, bool use_kd_tree) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

  solve_intersection<shape_type>(x, y, width, height, by_type_data, shapes,
                                 world_space_eyes, world_space_directions,
                                 ignores, disables, is_first, use_kd_tree);
}

template <scene::Shape shape_type>
inline void solve_intersections_cpu(
    unsigned width, unsigned height, ByTypeDataRef by_type_data,
    const scene::ShapeData *shapes, const Eigen::Vector3f *world_space_eyes,
    const Eigen::Vector3f *world_space_directions, const unsigned *ignores,
    const uint8_t *disables, bool is_first, bool use_kd_tree) {
  for (unsigned x = 0; x < width; x++) {
    for (unsigned y = 0; y < height; y++) {
      solve_intersection<shape_type>(x, y, width, height, by_type_data, shapes,
                                     world_space_eyes, world_space_directions,
                                     ignores, disables, is_first, use_kd_tree);
    }
  }
}

template <typename... T>
__host__ __device__ void minimize_intersections(
    unsigned index, unsigned size, bool is_first, const uint8_t *disables,
    thrust::optional<BestIntersectionNormalUV> *first, T... rest) {
  if (index >= size) {
    return;
  }
  if (!is_first && disables[index]) {
    return;
  }

  // needed????
  auto new_first = optional_min(first[index], rest[index]...);
  first[index] = new_first;
}

template <typename... T>
__global__ void minimize_all_intersections(unsigned size, bool is_first,
                                           const uint8_t *disables, T... rest) {
  unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  minimize_intersections(index, size, is_first, disables, rest...);
}

template <typename... T>
void minimize_all_intersections_cpu(unsigned size, bool is_first,
                                    const uint8_t *disables, T... rest) {
  for (unsigned index = 0; index < size; index++) {
    minimize_intersections(index, size, is_first, disables, rest...);
  }
}
} // namespace detail
} // namespace ray
