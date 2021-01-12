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

template <typename T>
concept IsPointer = std::is_pointer_v<T>;

template <typename T>
concept GetPtr = requires(T &&t) {
  typename GetPtrImpl<T>;
  { GetPtrImpl<T>::get(std::forward<T>(t)) } -> IsPointer;
};

template <GetPtr T> constexpr auto get_ptr(T &&t) {
  return GetPtrImpl<T>::get(std::forward<T>(t));
}

template <GetPtr T>
using PointerElemType =
    std::remove_reference_t<decltype(*get_ptr(std::declval<T>()))>;

template <typename T, typename Elem>
concept GetPtrForElem = requires(T &&t) {
  requires GetPtr<T>;
  { get_ptr(t) } -> std::convertible_to<Elem *>;
};
