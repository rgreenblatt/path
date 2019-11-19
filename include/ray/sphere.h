#pragma once

#include "ray/ray_utils.h"

namespace ray {
namespace detail {
template <bool get_normals> struct sphere_converter;

template <> struct sphere_converter<true> {
  HOST_DEVICE static auto get_converter(const Eigen::Vector3f &point,
                                        const Eigen::Vector3f &direction,
                                        const bool &texture_map) {
    return [&](float v) {
      // intersection is normal
      Eigen::Vector3f intersection = (point + direction * v).normalized();

      return IntersectionNormalUV(
          v, intersection,
          texture_map ? Eigen::Array2f(get_theta_div_uv(intersection),
                                       0.5f - std::asin(intersection.y()) /
                                                  static_cast<float>(M_PI))
                      : Eigen::Array2f(0));
    };
  }
};

template <> struct sphere_converter<false> {
  HOST_DEVICE static auto get_converter(const Eigen::Vector3f &,
                                        const Eigen::Vector3f &, bool) {
    return [](float v) { return v; };
  }
};

template <bool get_normals>
__host__ __device__ auto solve_sphere(const Eigen::Vector3f &point,
                                      const Eigen::Vector3f &direction,
                                      bool texture_map) {
  // R = 1/2
  // x^2 + y^2 + z^2 - 1 / 4 = 0
  // a = d_x^2 + d_y^2 + d_z^2
  // b = 2 * d_x * p_x + 2 * d_y * p_y + 2 * d_z * p_z
  // c = p_x^2 + p_y^2 + p_z^2 - 0.25
  float a = direction.squaredNorm();
  float b = 2.0f * point.cwiseProduct(direction).sum();
  float c = point.squaredNorm() - 0.25f;

  return optional_map(quadratic_formula(a, b, c),
                      sphere_converter<get_normals>::get_converter(
                          point, direction, texture_map));
}
} // namespace detail
} // namespace ray
