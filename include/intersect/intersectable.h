#pragma once

#include "intersect/intersection.h"
#include "intersect/ray.h"
#include "intersect/triangle.h"
#include "lib/concepts.h"
#include "lib/span.h"

#include <concepts>

namespace intersect {
template <typename T> struct IntersectableImpl;

// needs to be separate to avoid error for some reason
template <typename Impl, typename T>
concept IntersectableChecker = requires(const Ray &ray, const T &t,
                                        Span<const Triangle> dummy) {
  { Impl::intersect(ray, t, dummy) }
  ->SpecializationOf<IntersectionOp>;
};

template <typename T> concept Intersectable = requires {
  typename IntersectableImpl<T>;
  IntersectableChecker<IntersectableImpl<T>, T>;
};

#if 1
template <typename T> struct IntersectableT : IntersectableImpl<T> {
  template <typename... Vals>
  using Intersection = std::decay_t<decltype(
      IntersectableT::intersect(std::declval<const Ray &>(),
                                std::declval<const T &>(),
                                std::declval<Vals>()...)
          ->info)>;
};
#endif
} // namespace intersect
