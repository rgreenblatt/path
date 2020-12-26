#pragma once

#include "lib/bgra.h"
#include "lib/cuda/utils.h"

#include <Eigen/Geometry>

#include <algorithm>
#include <string_view>
#include <tuple>

namespace printf_dbg {
namespace detail {
template <typename Iter>
constexpr size_t copy_in_n_times(Iter format_iter, std::string_view s,
                                 size_t times) {
  size_t offset = 0;
  for (size_t i = 0; i < times; ++i) {
    std::copy(s.begin(), s.end(), format_iter);

    offset += s.size();
  }

  return offset;
}

// not ideal imo, some clean up is possible for sure...
template <typename Iter, typename T>
HOST_DEVICE inline auto debug_value(Iter format_iter, const T &val) {
  auto handle_vals = [&](std::string_view format, size_t times,
                         const auto &...vals) {
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
  } else if constexpr (std::is_pointer_v<T>) {
    return handle_vals("%p\n", 1, reinterpret_cast<const void *>(val));
  } else if constexpr (std::is_same<typename std::decay_t<T>, uint8_t>::value) {
    return handle_vals("%u\n", 1, val);
  } else if constexpr (std::is_same<typename std::decay_t<T>, float>::value) {
    return handle_vals("%g\n", 1, val);
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

  std::copy(initial_format.begin(), initial_format.end(),
            format_buffer.begin());

  const long to_remove =
      static_cast<long>(file_name.size()) - max_file_name_len;

  file_name.remove_prefix(std::max(to_remove, 0l));
  const auto [format_len, args] =
      debug_value(format_buffer.begin() + initial_format.size(), val);
  *(format_buffer.data() + (initial_format.size() + format_len)) = '\0';
  std::apply(
      [&](const auto &...additional) {
        printf(format_buffer.data(), to_remove < 0 ? "" : "..",
               file_name.data(), line_number, func.data(), var_name.data(),
               additional...);
      },
      args);

  return val;
}
} // namespace detail
} // namespace printf_dbg

#define printf_dbg(...)                                                        \
  printf_dbg::detail::debug_print(__FILE__, __LINE__, __func__, __VA_ARGS__,   \
                                  #__VA_ARGS__)
