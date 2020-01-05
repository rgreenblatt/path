#pragma once

#include "lib/span.h"

#include <thrust/device_vector.h>

#include <vector>

namespace ray {
namespace detail {
template <typename T> T *to_ptr(thrust::device_vector<T> &vec) {
  return thrust::raw_pointer_cast(vec.data());
}

template <typename T> const T *to_ptr(const thrust::device_vector<T> &vec) {
  return thrust::raw_pointer_cast(vec.data());
}

template <typename T, typename A> T *to_ptr(std::vector<T, A> &vec) {
  return vec.data();
}

template <typename T, typename A> T *to_thrust_iter(std::vector<T, A> &vec) {
  return to_ptr(vec);
}

template <typename T, typename A>
const T *to_thrust_iter(const std::vector<T, A> &vec) {
  return to_ptr(vec);
}

template <typename T> auto to_thrust_iter(thrust::device_vector<T> &vec) {
  return vec.begin();
}

template <typename T> auto to_thrust_iter(const thrust::device_vector<T> &vec) {
  return vec.begin();
}

template <typename T> Span<T> to_span(thrust::device_vector<T> &vec) {
  return Span(thrust::raw_pointer_cast(vec.data()), vec.size());
}

template <typename T>
Span<const T> to_const_span(const thrust::device_vector<T> &vec) {
  return Span(thrust::raw_pointer_cast(vec.data()), vec.size());
}

template <typename T, typename A> Span<T> to_span(std::vector<T, A> &vec) {
  return Span(vec.data(), vec.size());
}

template <typename T, typename A>
Span<const T> to_const_span(const std::vector<T, A> &vec) {
  return Span(vec.data(), vec.size());
}

template <typename T> const T *to_ptr(const std::vector<T> &vec) {
  return vec.data();
}
} // namespace detail
} // namespace ray
