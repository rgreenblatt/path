#pragma once

#include "lib/optional.h"
#include "ray/detail/intersection/intersection.h"

namespace ray {
namespace detail {
namespace intersection {
namespace shapes {
constexpr thrust::optional<float>
option_if_negative(thrust::optional<float> v) {
  return v.and_then([](const float v) { return make_optional(v >= 0, v); });
}

template <typename... T>
constexpr thrust::optional<float> optional_positive_min(T... values) {
  return optional_min(option_if_negative(values)...);
}

template <bool normal_and_uv>
using IntersectionOp = typename thrust::optional<Intersection<normal_and_uv>>;

// needs to be approx > 1 pixel at normal resolution
constexpr float epsilon = 1e-5;

template <typename F>
constexpr thrust::optional<float>
quadratic_formula(const float a, const float b, const float c, const F &check) {
  float determinant = b * b - 4 * a * c;
  if (determinant >= -epsilon) {
    determinant = determinant < 0 ? 0 : determinant;
    const auto get_sol = [&](const bool sgn) {
      const float sgn_v = sgn ? 1.0f : -1.0f;
      return (-b + sgn_v * std::sqrt(determinant)) / (2 * a);
    };

    return optional_positive_min(check(get_sol(true)), check(get_sol(false)));
  } else {
    return thrust::nullopt;
  }
}

constexpr thrust::optional<float>
quadratic_formula(const float a, const float b, const float c) {
  return quadratic_formula(a, b, c, [](const auto &a) { return a; });
}

HOST_DEVICE inline UVPosition uv_square_face(UVPosition vec,
                                             thrust::optional<int> positive) {
  if (positive.has_value()) {
    vec[*positive] *= -1.0f;
  }

  return vec + 0.5f;
}

template <bool normal_and_uv, bool is_top>
HOST_DEVICE inline IntersectionOp<normal_and_uv>
cap_sol(const Eigen::Vector3f &point, const Eigen::Vector3f &direction,
        bool texture_map) {
  float sol_v = ((is_top ? 0.5f : -0.5f) - point.y()) / direction.y();

  const Eigen::Vector2f x_z_intersection =
      sol_v * Eigen::Array2f(direction.x(), direction.z()) +
      Eigen::Array2f(point.x(), point.z());
  // x^2 + z^2 <= R^2
  // x^2 + z^2 <= 0.25
  const bool within_cap =
      sol_v >= 0 && x_z_intersection.squaredNorm() <= 0.25f + epsilon;

  return make_optional(
      within_cap, invoke([&] {
        if constexpr (normal_and_uv) {
          return IntersectionNormalUV(
              sol_v, Eigen::Vector3f(0, is_top ? 1 : -1, 0),
              texture_map
                  ? uv_square_face(x_z_intersection, make_optional(!is_top, 1))
                  : UVPosition());

        } else {
          return sol_v;
        }
      }));
}

HOST_DEVICE inline thrust::optional<float>
height_check(float t, const Eigen::Vector3f &point,
             const Eigen::Vector3f &direction) {
  return make_optional(std::abs(point.y() + t * direction.y()) < 0.5f + epsilon,
                       t);
}

HOST_DEVICE inline float get_theta_div_uv(const Eigen::Vector3f &unit_vec) {
  float theta_div = std::atan2(unit_vec.z(), unit_vec.x()) /
                    (2.0f * static_cast<float>(M_PI));

  return (theta_div < 0 ? 0 : 1) - theta_div;
}
} // namespace shapes
} // namespace intersection
} // namespace detail
} // namespace ray
