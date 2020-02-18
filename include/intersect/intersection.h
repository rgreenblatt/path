#pragma once

#include "lib/cuda/utils.h"

#include <thrust/optional.h>

namespace intersect {
template <typename Info> struct Intersection {
  float intersection_dist;
  Info info;
};

template <typename Info>
HOST_DEVICE inline bool operator<(const Intersection<Info> &lhs,
                                  const Intersection<Info> &rhs) {
  return lhs.intersection_dist < rhs.intersection_dist;
}

template <typename Info>
HOST_DEVICE inline bool operator>(const Intersection<Info> &lhs,
                                  const Intersection<Info> &rhs) {
  return operator<(rhs, lhs);
}

template <typename Info>
HOST_DEVICE inline bool operator<=(const Intersection<Info> &lhs,
                                   const Intersection<Info> &rhs) {
  return !operator>(lhs, rhs);
}

template <typename Info>
HOST_DEVICE inline bool operator>=(const Intersection<Info> &lhs,
                                   const Intersection<Info> &rhs) {
  return !operator<(lhs, rhs);
}

template <typename Info>
using IntersectionOp = thrust::optional<Intersection<Info>>;

struct TriangleLocation {
  unsigned triangle_idx;
};

struct GlobalLocation {
  unsigned mesh_idx;
  unsigned triangle_idx;
};
} // namespace intersect
