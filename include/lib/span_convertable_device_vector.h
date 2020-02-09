#pragma once

#include "lib/device_vector.h"
#include "lib/span_convertable.h"

template <typename T> class SpanConvertable<const DeviceVector<T>> {
public:
  constexpr static const T *ptr(const DeviceVector<T> &v) {
    return thrust::raw_pointer_cast(v.data());
  }

  constexpr static std::size_t size(const DeviceVector<T> &v) {
    return v.size();
  }
};

template <typename T> class SpanConvertable<DeviceVector<T>> {
public:
  constexpr static T *ptr(DeviceVector<T> &v) {
    return thrust::raw_pointer_cast(v.data());
  }

  constexpr static std::size_t size(const DeviceVector<T> &v) {
    return v.size();
  }
};
