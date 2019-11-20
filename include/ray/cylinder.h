#pragma once

#include "ray/ray_utils.h"

namespace ray {
namespace detail {
template <bool normal_and_uv> struct cylinder_body_converter;

template <> struct cylinder_body_converter<false> {
  HOST_DEVICE static auto get_converter(const Eigen::Vector3f &,
                                        const Eigen::Vector3f &, bool) {
    return [&](const float v) { return v; };
  }
};

template <> struct cylinder_body_converter<true> {
  HOST_DEVICE static auto get_converter(const Eigen::Vector3f &point,
                                        const Eigen::Vector3f &direction,
                                        const bool &texture_map) {
    return [&](float v) {
      const Eigen::Vector3f intersection = point + direction * v;
      const auto normal =
          Eigen::Vector3f(intersection.x(), 0, intersection.z());

      return IntersectionNormalUV(
          v, normal,
          texture_map ? Eigen::Array2f(get_theta_div_uv(intersection),
                                       -intersection.y() + 0.5f)
                      : Eigen::Array2f(0));
    };
  }
};

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

  const auto body_sol =
      optional_map(optional_and_then(quadratic_formula(a, b, c),
                                     [&](float v) {
                                       return height_check(v, point, direction);
                                     }),
                   cylinder_body_converter<normal_and_uv>::get_converter(
                       point, direction, texture_map));

  const auto bottom_cap_sol =
      cap_sol<normal_and_uv, false>(point, direction, texture_map);
  const auto top_cap_sol =
      cap_sol<normal_and_uv, true>(point, direction, texture_map);

  return optional_min(body_sol, bottom_cap_sol, top_cap_sol);
}
} // namespace detail
} // namespace ray
