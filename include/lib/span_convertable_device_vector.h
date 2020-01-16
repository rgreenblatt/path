#pragma once

#include "lib/span_convertable.h"

#include <thrust/device_vector.h>

template <typename T, typename A>
class SpanConvertable<const thrust::device_vector<T, A>> {
public:
  constexpr static const T *ptr(const thrust::device_vector<T, A> &v) {
    return thrust::raw_pointer_cast(v.data());
  }

  constexpr static std::size_t size(const thrust::device_vector<T, A> &v) {
    return v.size();
  }
};

template <typename T, typename A>
class SpanConvertable<thrust::device_vector<T, A>> {
public:
  constexpr static T *ptr(thrust::device_vector<T, A> &v) {
    return thrust::raw_pointer_cast(v.data());
  }

  constexpr static std::size_t size(const thrust::device_vector<T, A> &v) {
    return v.size();
  }
};
