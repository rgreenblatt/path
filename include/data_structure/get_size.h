#pragma once

#include "data_structure/detail/vector_like.h"

#include <concepts>

template <typename T> struct GetSizeImpl;

template <detail::VectorLike T> struct GetSizeImpl<T> {
  static auto get(T &&v) { return v.size(); }
};

template <typename T> concept GetSize = requires(T &&t) {
  typename GetSizeImpl<T>;
  { GetSizeImpl<T>::get(std::forward<T>(t)) }
  ->std::convertible_to<std::size_t>;
};

template <GetSize T> struct GetSizeT : GetSizeImpl<T> {};
