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
  requires std::copyable<typename T::InfoType>;
  { t.intersect(ray) }
  ->DecaysTo<IntersectionOp<typename T::InfoType>>;
};

struct MockIntersectable : MockNoRequirements {
  struct InfoType : MockCopyable {};

  constexpr IntersectionOp<InfoType> intersect(const Ray &) const {
    return nullopt_value;
  }
};
} // namespace intersect
