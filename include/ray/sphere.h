#pragma once

#include "ray/ray_utils.h"

namespace ray {
namespace detail {
template <bool normal_and_uv>
__host__ __device__ IntersectionOp<normal_and_uv>
solve_sphere(const Eigen::Vector3f &point, const Eigen::Vector3f &direction,
             bool texture_map) {
  // R = 1/2
  // x^2 + y^2 + z^2 - 1 / 4 = 0
  // a = d_x^2 + d_y^2 + d_z^2
  // b = 2 * d_x * p_x + 2 * d_y * p_y + 2 * d_z * p_z
  // c = p_x^2 + p_y^2 + p_z^2 - 0.25
  float a = direction.squaredNorm();
  float b = 2.0f * point.cwiseProduct(direction).sum();
  float c = point.squaredNorm() - 0.25f;
      
  return optional_map(quadratic_formula(a, b, c), [&](float v) {
    if constexpr (normal_and_uv) {
      // intersection is normal
      Eigen::Vector3f intersection = (point + direction * v).normalized();

      return IntersectionNormalUV(
          v, intersection,
          texture_map ? Eigen::Array2f(get_theta_div_uv(intersection),
                                       0.5f - std::asin(intersection.y()) /
                                                  static_cast<float>(M_PI))
                      : Eigen::Array2f(0));
    } else {
      return v;
    }
  });
}
} // namespace detail
} // namespace ray
