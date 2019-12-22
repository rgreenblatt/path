#pragma once

#include "ray/projection.h"

namespace ray {
namespace detail {
inline HOST_DEVICE std::tuple<Eigen::Array2f, float>
project_point(const Eigen::Vector3f &point, const TriangleProjector &projector,
              bool flip_x = false, bool flip_y = false) {
  Eigen::Array2f projected_point;
  float dist;

  projector.visit([&](const auto &v) {
    using T = std::decay_t<decltype(v)>;
    if constexpr (std::is_same<T, DirectionPlane>::value) {
      auto [p, d] = v.get_intersection_point(point);
      projected_point = p;
      dist = d;
    } else {
      auto projected = apply_projective(point, v);
      projected_point = projected.template head<2>();
      dist = projected.z();
    }
  });

  if (flip_x) {
    projected_point.x() *= -1.0f;
  }

  if (flip_y) {
    projected_point.y() *= -1.0f;
  }

  return std::make_tuple(projected_point, dist);
}
} // namespace detail
} // namespace ray
