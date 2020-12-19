#pragma once

#include "intersect/intersection.h"
#include "intersect/ray.h"

#include <concepts>

namespace intersect {
template <typename T>
concept Intersectable = requires(const T &t, const Ray &ray) {
  typename T::InfoType;
  requires std::copyable<typename T::InfoType>;
  { t.intersect(ray) }
  ->std::convertible_to<IntersectionOp<typename T::InfoType>>;
};

struct MockIntersectable {
  using InfoType = std::tuple<>;

  constexpr IntersectionOp<InfoType> intersect(const Ray&) const {
    return thrust::nullopt;
  }
};
}
