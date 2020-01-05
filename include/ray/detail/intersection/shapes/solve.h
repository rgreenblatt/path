#pragma once

#include "ray/detail/intersection/shapes/cone.h"
#include "ray/detail/intersection/shapes/cube.h"
#include "ray/detail/intersection/shapes/cylinder.h"
#include "ray/detail/intersection/shapes/sphere.h"
#include "scene/shape.h"
#include "scene/shape_data.h"

namespace ray {
namespace detail {
namespace intersection {
namespace shapes {
template <bool normal_and_uv>
inline HOST_DEVICE IntersectionOp<normal_and_uv>
solve_type(scene::Shape shape_type, const Eigen::Vector3f &point,
           const Eigen::Vector3f &direction, bool texture_map) {
  switch (shape_type) {
  case scene::Shape::Sphere:
    return solve_sphere<normal_and_uv>(point, direction, texture_map);
  case scene::Shape::Cylinder:
    return solve_cylinder<normal_and_uv>(point, direction, texture_map);
  case scene::Shape::Cube:
    return solve_cube<normal_and_uv>(point, direction, texture_map);
  case scene::Shape::Cone:
    return solve_cone<normal_and_uv>(point, direction, texture_map);
  }
}

// should this move somewhere for generality????
template <bool normal_and_uv>
inline HOST_DEVICE thrust::optional<BestIntersectionGeneral<normal_and_uv>>
get_intersection(Span<const scene::ShapeData> shapes, const unsigned shape_idx,
                 const Eigen::Vector3f &world_space_eye,
                 const Eigen::Vector3f &world_space_direction) {
  const auto &shape = shapes[shape_idx];
  const auto object_space_eye = shape.get_world_to_object() * world_space_eye;
  const auto object_space_direction =
      shape.get_world_to_object().linear() * world_space_direction;

  return optional_map(
      solve_type<normal_and_uv>(
          shape.get_shape(), object_space_eye, object_space_direction,
          normal_and_uv && shape.get_material().texture_data.has_value()),
      [&](const auto &value) {
        return BestIntersectionGeneral<normal_and_uv>(value, shape_idx);
      });
}
} // namespace shapes
} // namespace intersection
} // namespace detail
} // namespace ray
