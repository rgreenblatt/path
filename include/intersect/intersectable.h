#pragma once

#include "intersect/intersection.h"
#include "intersect/ray.h"
#include "meta/decays_to.h"
#include "meta/mock.h"

#include <concepts>

namespace intersect {
template <typename T>
concept Intersectable = requires(const T &t, const Ray &ray) {
  typename T::InfoType;
  requires std::semiregular<typename T::InfoType>;
  { t.intersect(ray) }
  ->DecaysTo<IntersectionOp<typename T::InfoType>>;
};

template <typename T, typename InfoType>
concept IntersectableWithInfoType = requires {
  requires Intersectable<T>;
  requires std::same_as<typename T::InfoType, InfoType>;
};

struct MockIntersectable : MockNoRequirements {
  struct InfoType : MockSemiregular {};

  constexpr IntersectionOp<InfoType> intersect(const Ray &) const {
    return nullopt_value;
  }
};

static_assert(Intersectable<MockIntersectable>);
} // namespace intersect
