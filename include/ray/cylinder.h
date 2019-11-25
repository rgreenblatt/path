#pragma once

#include "ray/ray_utils.h"

namespace ray {
namespace detail {
template <bool normal_and_uv>
HOST_DEVICE IntersectionOp<normal_and_uv>
solve_cylinder(const Eigen::Vector3f &point, const Eigen::Vector3f &direction,
               bool texture_map) {
  // x^2 + z^2 = R^2
  // a:  d_x**2 + d_z**2
  // b:  2*d_x*p_x + 2*d_z*p_z
  // c:  p_x**2 + p_z**2 - 0.25
  const Eigen::Vector2f direction_xz(direction.x(), direction.z());
  const Eigen::Vector2f point_xz(point.x(), point.z());
  const float a = direction_xz.squaredNorm();
  const float b = 2.0f * (direction_xz.cwiseProduct(point_xz)).sum();
  const float c = point_xz.squaredNorm() - 0.25f;

  const auto body_sol = optional_map(
      quadratic_formula(a, b, c).and_then(
          [&](float v) { return height_check(v, point, direction); }),
      [&](float v) {
        if constexpr (normal_and_uv) {
          const Eigen::Vector3f intersection = point + direction * v;
          const auto normal =
              Eigen::Vector3f(intersection.x(), 0, intersection.z());

          return IntersectionNormalUV(
              v, normal,
              texture_map ? UVPosition(get_theta_div_uv(intersection),
                                           -intersection.y() + 0.5f)
                          : UVPosition());
        } else {
          return v;
        }
      });

  const auto bottom_cap_sol =
      cap_sol<normal_and_uv, false>(point, direction, texture_map);
  const auto top_cap_sol =
      cap_sol<normal_and_uv, true>(point, direction, texture_map);

  return optional_min(body_sol, bottom_cap_sol, top_cap_sol);
}
} // namespace detail
} // namespace ray
