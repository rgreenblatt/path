#pragma once

#include "lib/cuda/utils.h"
#include "meta/mock.h"

#ifdef __CUDACC__
#include <thrust/device_vector.h>
#endif

#ifdef __CUDACC__
namespace device_vector {
namespace detail {
template <typename T>
struct uninitialized_allocator : thrust::device_allocator<T> {
  HOST_DEVICE void construct(T *) {
    // no-op
  }
};
} // namespace detail
} // namespace device_vector


template <typename T>
using DeviceVector = thrust::device_vector<T,
      device_vector::detail::uninitialized_allocator<T>>;
#else
// Shouldn't be used, just a placeholder for cpu builds
template<typename T>
struct DeviceVector : MockNoRequirements {};
#endif
