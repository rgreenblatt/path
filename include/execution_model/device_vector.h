#pragma once

#ifdef __CUDACC__
#include "lib/cuda/utils.h"

#include <thrust/device_vector.h>

#include <type_traits>

namespace device_vector {
namespace detail {
template <typename T>
struct uninitialized_allocator : thrust::device_allocator<T> {
  HOST_DEVICE void construct(T *) {
    // no-op
  }
};

template<typename T>
struct DeviceVectorImplT {
  using Type = thrust::device_vector<T>;
};

// Note that right now Eigen types aren't either of these - this is
// a bit unfortunate...
template<typename T>
requires(std::is_trivially_default_constructible_v<T> ||
         std::is_trivially_copyable_v<T>) struct DeviceVectorImplT<T> {
  using Type =
      thrust::device_vector<T, uninitialized_allocator<T>>;
};
} // namespace detail
} // namespace device_vector

template <typename T>
using DeviceVector = typename device_vector::detail::DeviceVectorImplT<T>::Type;
#else
#include "meta/mock.h"

// Shouldn't be used, just a placeholder for cpu builds
template <typename T> struct DeviceVector : MockNoRequirements {};
#endif
