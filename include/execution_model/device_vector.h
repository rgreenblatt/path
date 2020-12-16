#pragma once

#include "data_structure/vector.h"
#include "lib/cuda/utils.h"

#include <thrust/device_vector.h>

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
using DeviceVector =
    thrust::device_vector<T, device_vector::detail::uninitialized_allocator<T>>;
static_assert(Vector<DeviceVector<int>>);
