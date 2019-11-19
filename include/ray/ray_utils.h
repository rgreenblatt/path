#pragma once

#include "lib/cuda_utils.h"

#include <Eigen/Core>
#include <optional>
#include <utility>

namespace ray {
namespace detail {
template <typename T, typename F,
          typename Ret = decltype(std::declval<F>()(std::declval<T>()))>
HOST_DEVICE inline Ret optional_and_then(const std::optional<T> &v,
                                         const F &f) {
  if (v.has_value()) {
    return f(*v);
  } else {
    return std::nullopt;
  }
}

template <typename T, typename F, typename Ret = decltype(std::declval<F>()())>
HOST_DEVICE inline Ret optional_or_else(const std::optional<T> &v, const F &f) {
  if (v.has_value()) {
    return v;
  } else {
    return f();
  }
}

template <typename T>
HOST_DEVICE inline std::optional<T> optional_or(const std::optional<T> &v,
                                                const std::optional<T> &e) {
  return optional_or_else(v, [&]() { return e; });
}

template <typename T, typename F,
          typename Ret = decltype(std::declval<F>()(std::declval<T>()))>
HOST_DEVICE inline std::optional<Ret> optional_map(const std::optional<T> &v,
                                                   const F &f) {
  if (v.has_value()) {
    return f(*v);
  } else {
    return std::nullopt;
  }
}

template <typename FFold, typename FBase, typename V>
HOST_DEVICE inline auto optional_fold(const FFold &, const FBase &f_base,
                                      const std::optional<V> &first) {
  return optional_map(first, f_base);
}

template <typename FFold, typename FBase, typename V, typename... T>
HOST_DEVICE inline auto optional_fold(const FFold &f_fold, const FBase &f_base,
                                      const std::optional<V> &first,
                                      const std::optional<T> &... rest) {
  const auto f_rest = optional_fold(f_fold, f_base, rest...);
  const decltype(f_rest) next = optional_and_then(first, [&](const auto &v) {
    if (f_rest.has_value()) {
      // make optional??
      return f_fold(*f_rest, v);
    } else {
      return std::make_optional(f_base(v));
    }
  });
  return optional_or(next, f_rest);
}

struct IntersectionNormalUV {
  float intersection;
  Eigen::Vector3f normal;
  Eigen::Array2f uv;

  HOST_DEVICE IntersectionNormalUV(float intersection,
                                   const Eigen::Vector3f &normal,
                                   const Eigen::Array2f &uv)
      : intersection(intersection), normal(normal), uv(uv) {}
};

template <typename... T>
HOST_DEVICE inline auto optional_min(std::optional<T>... values) {
  return optional_fold(
      [](const auto &a, const auto &b) {
        return std::make_optional(std::min(a, b));
      },
      [](const auto &a) { return a; }, values...);
}

template <typename T>
HOST_DEVICE inline std::optional<T> make_optional(bool condition, const T &v) {
  if (condition) {
    return v;
  } else {
    return std::nullopt;
  }
}

HOST_DEVICE inline std::optional<float>
option_if_negative(std::optional<float> v) {
  return optional_and_then(
      v, [](const float v) { return make_optional(v >= 0, v); });
}

template <typename... T>
HOST_DEVICE inline std::optional<float> optional_positive_min(T... values) {
  return optional_min(option_if_negative(values)...);
}

template <typename F>
HOST_DEVICE inline std::optional<float>
quadratic_formula(const float a, const float b, const float c, const F &check) {
  float determinant = b * b - 4 * a * c;
  if (determinant >= -std::numeric_limits<float>::epsilon()) {
    determinant = determinant < 0 ? 0 : determinant;
    const auto get_sol = [&](const bool sgn) {
      const float sgn_v = sgn ? 1.0f : -1.0f;
      return (-b + sgn_v * std::sqrt(determinant)) / (2 * a);
    };

    return optional_positive_min(check(get_sol(true)), check(get_sol(false)));
  } else {
    return std::nullopt;
  }
}

HOST_DEVICE inline std::optional<float>
quadratic_formula(const float a, const float b, const float c) {
  return quadratic_formula(a, b, c, [](const auto &a) { return a; });
}

HOST_DEVICE inline Eigen::Array2f uv_square_face(Eigen::Array2f vec,
                                                 std::optional<int> positive) {
  if (positive.has_value()) {
    vec[*positive] *= -1.0f;
  }

  return vec + 0.5f;
}

template <bool get_normals, bool is_top> struct convert_cap;

template <bool is_top> struct convert_cap<false, is_top> {
  HOST_DEVICE static float convert(float t, const Eigen::Vector2f &, bool) {
    return t;
  }
};

template <bool is_top> struct convert_cap<true, is_top> {
  HOST_DEVICE static IntersectionNormalUV
  convert(float t, const Eigen::Vector2f &location, bool texture_map) {
    return IntersectionNormalUV(
        t, Eigen::Vector3f(0, is_top ? 1 : -1, 0),
        texture_map ? uv_square_face(location, make_optional(!is_top, 1))
                    : Eigen::Array2f(0));
  }
};

template <bool get_normals, bool is_top>
HOST_DEVICE inline auto cap_sol(const Eigen::Vector3f &point,
                                const Eigen::Vector3f &direction,
                                bool texture_map) {
  float sol_v = ((is_top ? 0.5f : -0.5f) - point.y()) / direction.y();

  const Eigen::Vector2f x_z_intersection =
      sol_v * Eigen::Array2f(direction.x(), direction.z()) +
      Eigen::Array2f(point.x(), point.z());
  // x^2 + z^2 <= R^2
  // x^2 + z^2 <= 0.25
  const bool within_cap =
      sol_v >= 0 && x_z_intersection.squaredNorm() <=
                        0.25f + std::numeric_limits<float>::epsilon();

  return make_optional(within_cap, convert_cap<get_normals, is_top>::convert(
                                       sol_v, x_z_intersection, texture_map));
}

HOST_DEVICE inline std::optional<float>
height_check(float t, const Eigen::Vector3f &point,
             const Eigen::Vector3f &direction) {
  return make_optional(std::abs(point.y() + t * direction.y()) <
                           0.5f + std::numeric_limits<float>::epsilon(),
                       t);
}

HOST_DEVICE inline float get_theta_div_uv(const Eigen::Vector3f &unit_vec) {
  float theta_div = std::atan2(unit_vec.z(), unit_vec.x()) /
                    (2.0f * static_cast<float>(M_PI));
  return (theta_div < 0 ? 0 : 1) - theta_div;
}
} // namespace detail
} // namespace ray
