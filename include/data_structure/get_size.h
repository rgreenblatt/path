#pragma once

#include "data_structure/detail/vector_like.h"
#include "lib/attribute.h"

#include <concepts>

template <typename T> struct GetSizeImpl;

template <detail::VectorLike T> struct GetSizeImpl<T> {
  ATTR_PURE_NDEBUG static constexpr auto get(const T &v) { return v.size(); }
};

template <typename T>
concept GetSize = requires(const T &t) {
  typename GetSizeImpl<T>;
  { GetSizeImpl<T>::get(t) } -> std::convertible_to<std::size_t>;
};

template <GetSize T> constexpr std::size_t get_size(const T &t) {
  return GetSizeImpl<T>::get(t);
}
