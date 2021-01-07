#pragma once

#include "lib/cuda/utils.h"
#include "meta/container_concepts.h"
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

// if we need to expand to more general type, we can use an
// uninitialized allocator for just TriviallyDestructable types
template <TriviallyDestructable T>
using DeviceVector = thrust::device_vector<T
// how much perf does this get?
#if 0
    , device_vector::detail::uninitialized_allocator<T>
#endif
                                           >;
#else
// Shouldn't be used, just a placeholder for cpu builds
template <typename T> struct DeviceVector : MockNoRequirements {};
#endif
