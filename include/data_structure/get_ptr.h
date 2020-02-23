#pragma once

#include "lib/concepts.h"

#include <thrust/device_vector.h>

#include <array>
#include <concepts>
#include <vector>

template <typename T> struct GetPtrImpl;

template <typename T>
    requires StdArraySpecialization<T> ||
    SpecializationOf<T, std::vector> struct GetPtrImpl<T> {
  static auto get(T &&v) { return v.data(); }
};

template <typename T>
requires SpecializationOf<T, thrust::device_vector> struct GetPtrImpl<T> {
  static auto get(T &&t) { return thrust::raw_pointer_cast(t.data()); }
};

template <typename T, typename Elem> concept GetPtr = requires(T &&t) {
  typename GetPtrImpl<T>;
  { GetPtrImpl<T>::get(std::forward<T>(t)) }
  ->std::convertible_to<Elem *>;
};

template <typename T, typename Elem>
requires GetPtr<T, Elem> struct GetPtrChecked {
  using type = GetPtrImpl<T>;
};

template <typename T, typename Elem>
using GetPtrT = typename GetPtrChecked<T, Elem>::type;
