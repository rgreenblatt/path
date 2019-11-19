#pragma once

#include "ray/cube.h"
#include "ray/cylinder.h"
#include "ray/ray_utils.h"
#include "ray/sphere.h"
#include "scene/shape.h"
#include "scene/shape_data.h"

#include <optional>

namespace ray {
namespace detail {

struct BestIntersection {
  float intersection;
  unsigned shape_index;

  __host__ __device__ BestIntersection(const float intersection,
                                       const unsigned shape_index)
      : intersection(intersection), shape_index(shape_index) {}
};

template <bool get_normals>
__host__ __device__ auto
solve_type(scene::Shape type, const Eigen::Vector3f &point,
           const Eigen::Vector3f &direction, bool texture_map) {
  switch (type) {
  case scene::Shape::Cube:
    return solve_cube<get_normals>(point, direction, texture_map);
  case scene::Shape::Sphere:
    return solve_sphere<get_normals>(point, direction, texture_map);
  case scene::Shape::Cylinder:
  default:
    return solve_cylinder<get_normals>(point, direction, texture_map);
  }
}

__global__ void
solve_intersections(unsigned width, unsigned height, unsigned num_shapes,
                    const scene::Transform &m_film_to_world,
                    const Eigen::Vector3f &world_space_eye,
                    const scene::ShapeData *shapes,
                    Eigen::Vector3f *world_space_directions,
                    std::optional<BestIntersection> *best_intersections,
                    scene::Shape type, std::optional<unsigned> ignore) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned shape_idx___ = blockIdx.z * blockDim.z + threadIdx.z; // TODO
  unsigned index = shape_idx___ + num_shapes * (x + y * width);
  
  if (index >= width * height * num_shapes) {
    return;
  }

  const Eigen::Vector3f camera_space_film_plane(
      (2.0f * static_cast<float>(x)) / static_cast<float>(width) - 1.0f,
      (-2.0f * static_cast<float>(y)) / static_cast<float>(height) + 1.0f, -1);

  const auto world_space_film_plane = m_film_to_world * camera_space_film_plane;
  // TODO maybe auto...
  Eigen::Vector3f world_space_direction = (world_space_film_plane - world_space_eye).normalized();

  auto &best = best_intersections[index];

  for (unsigned shape_idx = 0; shape_idx < num_shapes; shape_idx++) {
    if (ignore && *ignore == shape_idx) {
      continue;
    }

    const auto &shape = shapes[shape_idx];
    const auto object_space_eye = shape.get_world_to_object() * world_space_eye;
    const auto object_space_direction =
        shape.get_world_to_object().linear() * world_space_direction;

    const auto to_best_intersection = [&](float value) -> BestIntersection {
      return BestIntersection(value, shape_idx);
    };

    std::optional<BestIntersection> shape_intersection =
        optional_map(solve_type<false>(type, object_space_eye,
                                       object_space_direction, false),
                     to_best_intersection);

    best = optional_fold(
        [&](BestIntersection next,
            BestIntersection previous) -> std::optional<BestIntersection> {
          return next.intersection < previous.intersection ? next : previous;
        },
        [](const auto &v) { return v; }, shape_intersection, best);
  }
}
} // namespace detail
} // namespace ray
