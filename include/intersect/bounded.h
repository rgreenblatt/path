#pragma once

#include "intersect/accel/aabb.h"
#include "lib/concepts.h"

#include <concepts>

namespace intersect {
template <typename T> struct BoundedImpl;

template <typename Impl, typename T>
concept BoundedChecker = requires(const T &t) {
  { BoundedImpl<T>::bounds(t) }
  ->std::convertible_to<accel::AABB>;
};

template <typename T> concept Bounded = requires(const T &t) {
  typename BoundedImpl<T>;
  BoundedChecker<BoundedImpl<T>, T>;
};

template <Bounded T> using BoundedT = BoundedImpl<T>;
} // namespace intersect
