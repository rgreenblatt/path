#pragma once

#include "lib/bgra_32.h"
#include "lib/cuda/utils.h"
#include "lib/float_rgb.h"

#include <Eigen/Geometry>
#include <boost/hana/string.hpp>

#include <algorithm>
#include <string_view>
#include <tuple>

namespace printf_dbg {
namespace detail {
template <unsigned n, typename T> constexpr auto rep(T v) {
  if constexpr (n == 0) {
    return BOOST_HANA_STRING("");
  } else {
    return v + rep<n - 1>(v);
  }
}

// not ideal imo, some clean up is possible for sure...
template <typename T> constexpr inline auto debug_value(const T &val) {
  auto handle_vals = [&](auto str, const auto &...vals) {
    return std::make_tuple(str, std::make_tuple(vals...));
  };

  if constexpr (std::is_same<typename std::decay_t<T>,
                             Eigen::Affine3f>::value) {
    return debug_value(val.matrix());
  } else if constexpr (std::is_same<typename std::decay_t<T>,
                                    Eigen::Matrix3f>::value) {
    return handle_vals(rep<3>(BOOST_HANA_STRING("x: %f, y: %f, z: %f\n")),
                       val(0, 0), val(0, 1), val(0, 2), val(1, 0), val(1, 1),
                       val(1, 2), val(2, 0), val(2, 1), val(2, 2));
  } else if constexpr (std::is_same<typename std::decay_t<T>,
                                    Eigen::Matrix4f>::value) {
    return handle_vals(
        rep<4>(BOOST_HANA_STRING("x: %f, y: %f, z: %f, w: %f\n")), val(0, 0),
        val(0, 1), val(0, 2), val(1, 0), val(1, 1), val(1, 2), val(2, 0),
        val(2, 1), val(2, 2), val(3, 0), val(3, 1), val(3, 2), val(3, 3));
  } else if constexpr (std::is_same<typename std::decay_t<T>,
                                    Eigen::Vector3f>::value ||
                       std::is_same<typename std::decay_t<T>,
                                    FloatRGB>::value) {
    return handle_vals(BOOST_HANA_STRING("x: %f, y: %f, z: %f\n"), val.x(),
                       val.y(), val.z());
  } else if constexpr (std::is_same<typename std::decay_t<T>, BGRA32>::value) {
    return handle_vals(BOOST_HANA_STRING("x: %u, y: %u, z: %u\n"), val.x(),
                       val.y(), val.z());
  } else if constexpr (std::is_same<typename std::decay_t<T>, char *>::value) {
    return handle_vals(BOOST_HANA_STRING("%s\n"), val);
  } else if constexpr (std::is_pointer_v<T>) {
    return handle_vals(BOOST_HANA_STRING("%p\n"),
                       reinterpret_cast<const void *>(val));
  } else if constexpr (std::is_same<typename std::decay_t<T>, uint8_t>::value) {
    return handle_vals(BOOST_HANA_STRING("%u\n"), val);
  } else if constexpr (std::is_same<typename std::decay_t<T>, float>::value) {
    return handle_vals(BOOST_HANA_STRING("%g\n"), val);
  } else if constexpr (std::is_same<typename std::decay_t<T>,
                                    unsigned>::value) {
    return handle_vals(BOOST_HANA_STRING("%u\n"), val);
  } else if constexpr (std::is_same<typename std::decay_t<T>,
                                    unsigned long>::value) {
    return handle_vals(BOOST_HANA_STRING("%lu\n"), val);
  } else if constexpr (std::is_same<typename std::decay_t<T>, bool>::value) {
    return handle_vals(BOOST_HANA_STRING("%s\n"), val ? "true" : "false");
  } else {
    static_assert(std::is_same<typename std::decay_t<T>, int>::value,
                  "type not yet handled");
    return handle_vals(BOOST_HANA_STRING("%d\n"), val);
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

constexpr auto combine() {
  return std::make_tuple(BOOST_HANA_STRING(""), std::make_tuple());
}

template <typename First, typename... Rest>
constexpr auto combine(const First &first, const Rest &...rest) {
  auto rec = combine(rest...);
  return std::make_tuple(std::get<0>(first) + std::get<0>(rec),
                         std::tuple_cat(std::get<1>(first), std::get<1>(rec)));
}

template <class T> constexpr std::string_view type_name() {
#ifndef __clang__
  static_assert(false);
#endif
  std::string_view p = __PRETTY_FUNCTION__;
  return std::string_view(p.data() + 34, p.size() - 34 - 1);
}

template <typename First, typename... Rest>
HOST_DEVICE inline decltype(auto)
debug_print(std::string_view file_name, int line_number, std::string_view func,
            std::string_view var_name, First &&ret_val, Rest &&...rest) {
  auto internal = [](const auto &...vals) {
    return combine(debug_value(vals)...);
  };

  auto [formatting, vals] = internal(ret_val, rest...);

  auto overall_formating =
      BOOST_HANA_STRING("[%s%s:%d %s] %s =\n") + formatting;

  const long max_file_name_len = 20;
  const long to_remove =
      static_cast<long>(file_name.size()) - max_file_name_len;
  file_name.remove_prefix(std::max(to_remove, 0l));

  std::apply(
      [&](const auto &...additional) {
        printf(overall_formating.c_str(), to_remove < 0 ? "" : "..",
               file_name.data(), line_number, func.data(), var_name.data(),
               additional...);
      },
      vals);

  return std::forward<First>(ret_val);
}
} // namespace detail
} // namespace printf_dbg

#define printf_dbg(...)                                                        \
  printf_dbg::detail::debug_print(__FILE__, __LINE__, __func__, #__VA_ARGS__,  \
                                  __VA_ARGS__)
