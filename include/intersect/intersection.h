#pragma once

#include "intersect/ray.h"
#include "lib/cuda/utils.h"

#include <compare>
#include <cmath>
#include <thrust/optional.h>

namespace intersect {
template <typename InfoType> struct Intersection {
  float intersection_dist;
  bool is_back_intersection;
  InfoType info;

  HOST_DEVICE Eigen::Vector3f intersection_point(const Ray &ray) const {
    assert(std::abs(ray.direction.norm() - 1.f) < 1e-6); // TODO (fix)
    return ray.origin + ray.direction * intersection_dist;
  }

  template <typename F> HOST_DEVICE auto map_info(F &&f) const {
    return Intersection<std::decay_t<decltype(f(info))>>{
        intersection_dist, is_back_intersection, f(info)};
  }
};

template <typename InfoType>
HOST_DEVICE inline auto operator<=>(const Intersection<InfoType> &lhs,
                                    const Intersection<InfoType> &rhs) {
  return lhs.intersection_dist <=> rhs.intersection_dist;
}

template <typename InfoType>
using IntersectionOp = thrust::optional<Intersection<InfoType>>;
} // namespace intersect
