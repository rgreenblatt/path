#pragma once

#include "intersect/accel/aabb.h"
#include "lib/concepts.h"

#include <concepts>

namespace intersect {
template <typename T> struct BoundedImpl;

template <typename T> concept Bounded = requires(const T &t) {
  typename BoundedImpl<T>;
  // TODO: constraint fails for unclear reasons
  /* { BoundedImpl<T>::bounds(t) } */
  /* ->std::convertible_to<accel::AABB>; */
};

template <Bounded T> using BoundedT = BoundedImpl<T>;
} // namespace intersect
