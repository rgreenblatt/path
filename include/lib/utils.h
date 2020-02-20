#pragma once

#include "lib/cuda/utils.h"

#include <Eigen/Core>

#include <string_view>

template <class F, class... Args>
constexpr decltype(auto) invoke(F &&f, Args &&... args) {
  return std::forward<F>(f)(std::forward<Args>(args)...);
}

template <class InputIt, class OutputIt>
constexpr OutputIt copy(InputIt first, InputIt last, OutputIt d_first) {
  while (first != last) {
    *d_first++ = *first++;
  }
  return d_first;
}

template <typename T> constexpr void swap_val(T &first, T &second) {
  T temp = first;
  first = second;
  second = temp;
}

HOST_DEVICE inline unsigned ceil_divide(unsigned l, unsigned r) {
  return (l + r - 1) / r;
};

HOST_DEVICE inline Eigen::Vector3f max_eigen_vec() {
  return Eigen::Vector3f(std::numeric_limits<float>::max(),
                         std::numeric_limits<float>::max(),
                         std::numeric_limits<float>::max());
}

HOST_DEVICE inline Eigen::Vector3f lowest_eigen_vec() {
  return Eigen::Vector3f(std::numeric_limits<float>::lowest(),
                         std::numeric_limits<float>::lowest(),
                         std::numeric_limits<float>::lowest());
}
