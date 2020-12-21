#pragma once

#include "data_structure/detail/vector_like.h"

#include <concepts>

template <typename T> struct GetPtrImpl;

template <detail::ArrayOrVector T> struct GetPtrImpl<T> {
  static auto get(T &&v) { return v.data(); }
};

template <detail::DeviceVector T> struct GetPtrImpl<T> {
  static auto get(T &&t) { return thrust::raw_pointer_cast(t.data()); }
};

template <typename T, typename Elem> concept GetPtr = requires(T &&t) {
  typename GetPtrImpl<T>;
  { GetPtrImpl<T>::get(std::forward<T>(t)) }
  ->std::convertible_to<Elem *>;
};

template <typename T, typename Elem>
requires GetPtr<T, Elem> struct GetPtrT : GetPtrImpl<T> {};
