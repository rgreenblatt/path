#pragma once

#include "meta/specialization_of.h"
#include "meta/std_array_specialization.h"

#include "execution_model/device_vector.h"

#include <debug/vector>
#include <vector>

namespace detail {
template <typename T>
concept ArrayOrVector =
    StdArraySpecialization<T> || SpecializationOf<T, __gnu_debug::vector> ||
    SpecializationOf<T, std::vector>;

#ifdef __CUDACC__
template <typename T>
concept IsDeviceVector = SpecializationOf<T, thrust::device_vector>;

static_assert(IsDeviceVector<DeviceVector<int>>);
#else
template <typename T>
concept IsDeviceVector = SpecializationOf<T, DeviceVector>;
#endif

template <typename T>
concept VectorLike = ArrayOrVector<T> || IsDeviceVector<T>;
} // namespace detail
