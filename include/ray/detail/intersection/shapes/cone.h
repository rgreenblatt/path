#pragma once

#include "ray/detail/intersection/shapes/utils.h"

namespace ray {
namespace detail {
namespace intersection {
namespace shapes {
template <bool normal_and_uv>
HOST_DEVICE auto solve_cone(const Eigen::Vector3f &point,
                            const Eigen::Vector3f &direction,
                            bool texture_map) {
  // R = 1/2
  // x^2 + z^2 - (-y/2 + 1/4)**2 = 0
  // a:  d_x^2 - 0.25*d_y^2 + d_z^2
  // b:  2*d_x*p_x - 0.5*d_y*p_y + 0.25*d_y + 2*d_z*p_z
  // c:  p_x^2 - 0.25*p_y^2 + 0.25*p_y + p_z^2 - 0.0625
  const auto dir_dir = direction.cwiseProduct(direction);
  const auto a = dir_dir.x() - 0.25f * dir_dir.y() + dir_dir.z();
  const auto dir_point = direction.cwiseProduct(point);
  const auto b = 2.0f * dir_point.x() - 0.5f * dir_point.y() +
                 2.0f * dir_point.z() + 0.25f * direction.y();
  const auto point_point = point.cwiseProduct(point);
  const auto c = point_point.x() - 0.25f * point_point.y() + 0.25f * point.y() +
                 point_point.z() - 0.0625f;

  // height check must be done on each solution...
  const auto body_sol = optional_map(
      quadratic_formula(
          a, b, c,
          [&](const auto &v) { return height_check(v, point, direction); }),
      [&](float v) {
        if constexpr (normal_and_uv) {
          const auto intersection = point + direction * v;
          const auto up =
              Eigen::Vector2f(intersection.x(), intersection.z()).norm() * 0.5f;
          const auto normal =
              Eigen::Vector3f(intersection.x(), up, intersection.z());

          return IntersectionNormalUV(
              v, normal,
              texture_map ? UVPosition(get_theta_div_uv(intersection),
                                       -intersection.y() + 0.5f)
                          : UVPosition(0));

        } else {
          return v;
        }
      });
  const auto bottom_cap_sol =
      cap_sol<normal_and_uv, false>(point, direction, texture_map);

  return optional_min(body_sol, bottom_cap_sol);
}
} // namespace shapes
} // namespace intersection
} // namespace detail
} // namespace ray
