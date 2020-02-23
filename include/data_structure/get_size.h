#pragma once

#include "lib/concepts.h"

#include <thrust/device_vector.h>

#include <array>
#include <concepts>
#include <vector>

template <typename T> struct GetSizeImpl;

template <typename T>
    requires StdArraySpecialization<T> || SpecializationOf<T, std::vector> ||
    SpecializationOf<T, thrust::device_vector> struct GetSizeImpl<T> {
  static auto get(T &&v) { return v.size(); }
};

template <typename T> concept GetSize = requires(T &&t) {
  typename GetSizeImpl<T>;
  { GetSizeImpl<T>::get(std::forward<T>(t)) }
  ->std::convertible_to<std::size_t>;
};

template <GetSize T> using GetSizeT = GetSizeImpl<T>;
