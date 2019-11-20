#pragma once

#include "ray/best_intersection.h"
#include "ray/cube.h"
#include "ray/cylinder.h"
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

  if (index >= width * height) {
    return;
  }

  const Eigen::Vector3f camera_space_film_plane(
      (2.0f * static_cast<float>(x)) / static_cast<float>(width) - 1.0f,
      (-2.0f * static_cast<float>(y)) / static_cast<float>(height) + 1.0f, -1);
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
           const Eigen::Vector3f &direction, bool texture_map) {
  switch (type) {
  case scene::Shape::Cube:
    return solve_cube<normal_and_uv>(point, direction, texture_map);
  case scene::Shape::Sphere:
    return solve_sphere<normal_and_uv>(point, direction, texture_map);
  case scene::Shape::Cylinder:
  default:
    return solve_cylinder<normal_and_uv>(point, direction, texture_map);
  }
}

template <bool normal_and_uv>
__host__
    __device__ inline thrust::optional<BestIntersectionGeneral<normal_and_uv>>
    get_shape_intersection(const scene::ShapeData *shapes,
                           const unsigned shape_idx,
                           const scene::Shape shape_type,
                           const Eigen::Vector3f &world_space_eye,
                           const Eigen::Vector3f &world_space_direction) {
  const auto &shape = shapes[shape_idx];
  const auto object_space_eye = shape.get_world_to_object() * world_space_eye;
  const auto object_space_direction =
      shape.get_world_to_object().linear() * world_space_direction;

  return optional_map(
      solve_type<normal_and_uv>(
          shape_type, object_space_eye, object_space_direction,
          normal_and_uv && shape.get_material().texture_map_index.has_value()),
      [&](const auto &value) {
        return BestIntersectionGeneral<normal_and_uv>(value, shape_idx);
      });
}

__host__ __device__ inline void solve_intersection(
    unsigned x, unsigned y, unsigned width, unsigned height, unsigned num_shape,
    unsigned start_shape, const scene::ShapeData *shapes,
    const Eigen::Vector3f *&world_space_eyes,
    const Eigen::Vector3f *world_space_directions, const unsigned *ignores,
    const uint8_t *disables,
    thrust::optional<BestIntersectionNormalUV> *best_intersections,
    const scene::Shape shape_type, bool is_first) {
  unsigned index = x + y * width;

  if (index >= width * height) {
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

  for (unsigned shape_idx = start_shape; shape_idx < start_shape + num_shape;
       shape_idx++) {
    if (!is_first && ignores[index] == shape_idx) {
      continue;
    }

    best = optional_min(
        get_shape_intersection<false>(shapes, shape_idx, shape_type,
                                      world_space_eye, world_space_direction),
        best);
  }

  if (best.has_value()) {
    best_normals_uv =
        get_shape_intersection<true>(shapes, best->shape_idx, shape_type,
                                     world_space_eye, world_space_direction);
  } else {
    best_normals_uv =
        thrust::optional<BestIntersectionNormalUV>(thrust::nullopt);
  }
}

__global__ void solve_intersections(
    unsigned width, unsigned height, unsigned num_shape, unsigned start_shape,
    const scene::ShapeData *shapes, const Eigen::Vector3f *world_space_eyes,
    const Eigen::Vector3f *world_space_directions, const unsigned *ignores,
    const uint8_t *disables,
    thrust::optional<BestIntersectionNormalUV> *best_intersections,
    const scene::Shape shape_type, bool is_first) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

  solve_intersection(x, y, width, height, num_shape, start_shape, shapes,
                     world_space_eyes, world_space_directions, ignores,
                     disables, best_intersections, shape_type, is_first);
}

inline void solve_intersections_cpu(
    unsigned width, unsigned height, unsigned num_shape, unsigned start_shape,
    const scene::ShapeData *shapes, const Eigen::Vector3f *world_space_eyes,
    const Eigen::Vector3f *world_space_directions, const unsigned *ignores,
    const uint8_t *disables,
    thrust::optional<BestIntersectionNormalUV> *best_intersections,
    const scene::Shape shape_type, bool is_first) {
  for (unsigned x = 0; x < width; x++) {
    for (unsigned y = 0; y < height; y++) {
      solve_intersection(x, y, width, height, num_shape, start_shape, shapes,
                         world_space_eyes, world_space_directions, ignores,
                         disables, best_intersections, shape_type, is_first);
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

  first[index] = optional_min(first[index], rest[index]...);
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
