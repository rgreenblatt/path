#pragma once

#include "data_structure/detail/vector_like.h"
#include "lib/attribute.h"

#include <concepts>

template <typename T> struct GetPtrImpl;

template <detail::ArrayOrVector T> struct GetPtrImpl<T> {
  ATTR_PURE_NDEBUG static constexpr auto get(T &&v) { return v.data(); }
};

#ifdef __CUDACC__
template <detail::IsDeviceVector T> struct GetPtrImpl<T> {
  ATTR_PURE_NDEBUG static auto get(T &&t) {
    return thrust::raw_pointer_cast(t.data());
  }
};
#endif

template <typename T, typename Elem>
concept GetPtr = requires(T &&t) {
  typename GetPtrImpl<T>;
  { GetPtrImpl<T>::get(std::forward<T>(t)) } -> std::convertible_to<Elem *>;
};

template <typename Elem, GetPtr<Elem> T> constexpr Elem *get_ptr(T &&t) {
  return GetPtrImpl<T>::get(std::forward<T>(t));
}
