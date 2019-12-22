#pragma once

#include "ray/intersect.h"
#include "ray/projection.h"

namespace ray {
namespace detail {
inline HOST_DEVICE Eigen::Array2f
project_point(const Eigen::Vector3f &point, const TriangleProjector &projector,
              bool flip_x = false, bool flip_y = false) {
  Eigen::Array2f projected_point;

  projector.visit([&](const auto &v) {
    using T = std::decay_t<decltype(v)>;
    if constexpr (std::is_same<T, DirectionPlane>::value) {
      projected_point = get_intersection_point(
          v.is_loc ? (v.loc_or_dir - point).eval() : v.loc_or_dir,
          v.projection_value, point, v.axis);
    } else {
      projected_point = apply_projective(point, v).template head<2>();
    }
  });

  if (flip_x) {
    projected_point.x() *= -1.0f;
  }

  if (flip_y) {
    projected_point.y() *= -1.0f;
  }

  return projected_point;
}
} // namespace detail
} // namespace ray
