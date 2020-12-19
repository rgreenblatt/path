#pragma once

#include "meta/concepts.h"

#include <thrust/device_vector.h>

#include <array>
#include <vector>

namespace detail {
template <typename T>
concept ArrayOrVector =
    StdArraySpecialization<T> || SpecializationOf<T, std::vector>;

template <typename T>
concept DeviceVector = SpecializationOf<T, thrust::device_vector>;

template <typename T> concept VectorLike = ArrayOrVector<T> || DeviceVector<T>;
} // namespace detail
