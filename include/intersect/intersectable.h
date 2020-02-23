#pragma once

#include "intersect/intersection.h"
#include "intersect/ray.h"
#include "lib/concepts.h"

#include <concepts>

namespace intersect {
template <typename T> struct IntersectableImpl;

template <typename T> concept Intersectable = requires {
  typename IntersectableImpl<T>;

  /* requires requires(const Ray &ray, const T &t) { */
  /*   { IntersectableImpl<T>::intersect(ray, t) } */
  /*   ->SpecializationOf<IntersectionOp>; */
  /* }; */
};

template <Intersectable T> struct IntersectableT : IntersectableImpl<T> {
  using Intersection = std::decay_t<decltype(
      IntersectableT::intersect(std::declval<const Ray &>(),
                                std::declval<const T &>())
          ->info)>;
};
} // namespace intersect
