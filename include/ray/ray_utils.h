#pragma once

#include "lib/cuda_utils.h"

#include <Eigen/Dense>

#include <thrust/optional.h>
#include <utility>

#include "lib/bgra.h"
#include "scene/shape_data.h"

namespace ray {
namespace detail {
// TODO
using UVPosition = Eigen::Array2f;

template <class F, class... Args>
constexpr decltype(auto) invoke(F &&f, Args &&... args) {
  return std::forward<F>(f)(std::forward<Args>(args)...);
}

template <typename T, typename F, typename Ret = decltype(std::declval<F>()())>
constexpr Ret optional_or_else(const thrust::optional<T> &v, const F &f) {
  if (v.has_value()) {
    return v;
  } else {
    return f();
  }
}

template <typename T>
constexpr thrust::optional<T> optional_or(const thrust::optional<T> &v,
                                          const thrust::optional<T> &e) {
  return optional_or_else(v, [&]() { return e; });
}

template <typename T, typename F,
          typename Ret = decltype(std::declval<F>()(std::declval<T>()))>
constexpr thrust::optional<Ret> optional_map(const thrust::optional<T> &v,
                                             const F &f) {
  if (v.has_value()) {
    return f(*v);
  } else {
    return thrust::nullopt;
  }
}

template <typename FFold, typename FBase, typename V>
constexpr auto optional_fold(const FFold &, const FBase &f_base,
                             const thrust::optional<V> &first) {
  return optional_map(first, f_base);
}

template <typename FFold, typename FBase, typename V, typename... T>
constexpr auto optional_fold(const FFold &f_fold, const FBase &f_base,
                             const thrust::optional<V> &first,
                             const thrust::optional<T> &... rest) {
  const auto f_rest = optional_fold(f_fold, f_base, rest...);

  const decltype(f_rest) next = first.and_then([&](const auto &v) {
    if (f_rest.has_value()) {
      // make optional??
      return f_fold(*f_rest, v);
    } else {
      return thrust::make_optional(f_base(v));
    }
  });

  return optional_or(next, f_rest);
}

struct IntersectionNormalUV {
  float intersection;
  Eigen::Vector3f normal;
  UVPosition uv;

  HOST_DEVICE IntersectionNormalUV(float intersection,
                                   const Eigen::Vector3f &normal,
                                   const UVPosition &uv)
      : intersection(intersection), normal(normal), uv(uv) {}

  constexpr bool operator<(const IntersectionNormalUV &rhs) const {
    return intersection < rhs.intersection;
  }
};

template <bool normal_and_uv>
using Intersection =
    typename std::conditional<normal_and_uv, IntersectionNormalUV, float>::type;

template <bool normal_and_uv>
using IntersectionOp = typename thrust::optional<Intersection<normal_and_uv>>;

template <typename... T>
constexpr auto optional_min(const thrust::optional<T> &... values) {
  return optional_fold(
      [](const auto &a, const auto &b) {
        return thrust::make_optional(std::min(a, b));
      },
      [](const auto &a) { return a; }, values...);
}

template <typename T>
constexpr thrust::optional<T> make_optional(bool condition, const T &v) {
  if (condition) {
    return thrust::make_optional(v);
  } else {
    return thrust::nullopt;
  }
}

constexpr thrust::optional<float>
option_if_negative(thrust::optional<float> v) {
  return v.and_then([](const float v) { return make_optional(v >= 0, v); });
}

template <typename... T>
constexpr thrust::optional<float> optional_positive_min(T... values) {
  return optional_min(option_if_negative(values)...);
}

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

template <bool normal_and_uv, bool is_top> struct convert_cap;

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
                    : UVPosition());
  }
};

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

  return make_optional(within_cap, convert_cap<normal_and_uv, is_top>::convert(
                                       sol_v, x_z_intersection, texture_map));
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

// constexpr
template <class InputIt, class OutputIt>
constexpr OutputIt copy(InputIt first, InputIt last, OutputIt d_first) {
  while (first != last) {
    *d_first++ = *first++;
  }
  return d_first;
}

template <typename Iter>
HOST_DEVICE constexpr size_t copy_in_n_times(Iter format_iter,
                                             std::string_view s, size_t times) {
  size_t offset = 0;
  for (size_t i = 0; i < times; ++i) {
    copy(s.begin(), s.end(), format_iter);

    offset += s.size();
  }

  return offset;
}

template <typename Iter, typename T>
HOST_DEVICE inline auto debug_value(Iter format_iter, const T &val) {
  auto handle_vals = [&](std::string_view format, size_t times,
                         const auto &... vals) {
    const size_t format_size = copy_in_n_times(format_iter, format, times);

    return std::make_tuple(format_size, std::make_tuple(vals...));
  };

  if constexpr (std::is_same<typename std::decay_t<T>,
                             Eigen::Affine3f>::value) {
    return debug_value(format_iter, val.matrix());
  } else if constexpr (std::is_same<typename std::decay_t<T>,
                                    Eigen::Matrix3f>::value) {
    return handle_vals("x: %f, y: %f, z: %f\n", 3, val(0, 0), val(0, 1),
                       val(0, 2), val(1, 0), val(1, 1), val(1, 2), val(2, 0),
                       val(2, 1), val(2, 2));
  } else if constexpr (std::is_same<typename std::decay_t<T>,
                                    Eigen::Matrix4f>::value) {
    return handle_vals("x: %f, y: %f, z: %f, w: %f\n", 4, val(0, 0), val(0, 1),
                       val(0, 2), val(1, 0), val(1, 1), val(1, 2), val(2, 0),
                       val(2, 1), val(2, 2), val(3, 0), val(3, 1), val(3, 2),
                       val(3, 3));
  } else if constexpr (std::is_same<typename std::decay_t<T>,
                                    Eigen::Vector3f>::value ||
                       std::is_same<typename std::decay_t<T>,
                                    Eigen::Array3f>::value) {
    return handle_vals("x: %f, y: %f, z: %f\n", 1, val.x(), val.y(), val.z());
  } else if constexpr (std::is_same<typename std::decay_t<T>, BGRA>::value) {
    return handle_vals("x: %u, y: %u, z: %u\n", 1, val.x(), val.y(), val.z());
  } else if constexpr (std::is_same<typename std::decay_t<T>, char *>::value) {
    return handle_vals("%s\n", 1, val);
  } else if constexpr (std::is_same<typename std::decay_t<T>, uint8_t>::value) {
    return handle_vals("%u\n", 1, val);
  } else if constexpr (std::is_same<typename std::decay_t<T>, float>::value) {
    return handle_vals("%f\n", 1, val);
  } else if constexpr (std::is_same<typename std::decay_t<T>,
                                    unsigned>::value) {
    return handle_vals("%u\n", 1, val);
  } else if constexpr (std::is_same<typename std::decay_t<T>,
                                    unsigned long>::value) {
    return handle_vals("%lu\n", 1, val);
  } else if constexpr (std::is_same<typename std::decay_t<T>, bool>::value) {
    return handle_vals("%s\n", 1, val ? "true" : "false");
  } else {
    static_assert(std::is_same<typename std::decay_t<T>, int>::value,
                  "type not yet handled");
    return handle_vals("%d\n", 1, val);
  }
}

#ifdef __CUDACC__
extern "C" {
__device__ inline size_t strlen(const char *v) {
  const char *s;

  for (s = v; *s; ++s) {
  }

  return (s - v);
}
}
#endif

template <typename T>
HOST_DEVICE inline auto debug_print(std::string_view file_name, int line_number,
                                    std::string_view func, const T &val,
                                    std::string_view var_name) {
  const long max_file_name_len = 20;

  std::array<char, 100> format_buffer;
  std::string_view initial_format = "[%s%s:%d %s] %s =\n";

  copy(initial_format.begin(), initial_format.end(), format_buffer.begin());

  const long to_remove =
      static_cast<long>(file_name.size()) - max_file_name_len;

  file_name.remove_prefix(std::max(to_remove, 0l));
  const auto [format_len, args] =
      debug_value(format_buffer.begin() + initial_format.size(), val);
  *(format_buffer.data() + (initial_format.size() + format_len)) = '\0';
  std::apply(
      [&](const auto &... additional) {
        printf(format_buffer.data(), to_remove < 0 ? "" : "..",
               file_name.data(), line_number, func.data(), var_name.data(),
               additional...);
      },
      args);

  return val;
}

#define printf_dbg(...)                                                        \
  debug_print(__FILE__, __LINE__, __func__, __VA_ARGS__, #__VA_ARGS__)

} // namespace detail
} // namespace ray
