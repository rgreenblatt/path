#pragma once

#include "lib/cuda/utils.h"

#include <thrust/optional.h>

namespace intersect {
template <typename InfoType> struct Intersection {
  float intersection_dist;
  bool is_back_intersection;
  InfoType info;
};

template <typename InfoType>
HOST_DEVICE inline auto operator<=>(const Intersection<InfoType> &lhs,
                                    const Intersection<InfoType> &rhs) {
  return lhs.intersection_dist <=> rhs.intersection_dist;
}

template <typename InfoType>
using IntersectionOp = thrust::optional<Intersection<InfoType>>;
} // namespace intersect
