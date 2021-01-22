#pragma once

#include "intersect/ray.h"
#include "lib/attribute.h"
#include "lib/cuda/utils.h"
#include "lib/optional.h"

#include <cmath>
#include <compare>

namespace intersect {
template <std::copyable InfoType> struct Intersection {
  float intersection_dist;
  bool is_back_intersection;
  [[no_unique_address]] InfoType info;

  ATTR_PURE_NDEBUG HOST_DEVICE Eigen::Vector3f
  intersection_point(const Ray &ray) const {
    return ray.origin + *ray.direction * intersection_dist;
  }

  template <typename F> HOST_DEVICE auto map_info(F &&f) const {
    return Intersection<std::decay_t<decltype(f(info))>>{
        intersection_dist, is_back_intersection, f(info)};
  }
};

template <typename InfoType>
ATTR_PURE HOST_DEVICE inline auto
operator<=>(const Intersection<InfoType> &lhs,
            const Intersection<InfoType> &rhs) {
  return lhs.intersection_dist <=> rhs.intersection_dist;
}

template <std::copyable InfoType>
using IntersectionOp = std::optional<Intersection<InfoType>>;
} // namespace intersect
