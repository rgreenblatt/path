#pragma once

#include "intersect/accel/loop_all/loop_all.h"
#include "intersect/accel/add_idx.h"

namespace intersect {
namespace accel {
namespace loop_all {
namespace detail {
template <Object O>
HOST_DEVICE inline IntersectionOp<typename Ref::IntersectableRef<O>::InfoType>
Ref::IntersectableRef<O>::intersect(const Ray &ray) const {
  IntersectionOp<InfoType> best_intersection;

  for (unsigned idx = 0; idx < ref.size; idx++) {
    best_intersection = optional_min(
        best_intersection, add_idx(objects[idx].intersect(ray), idx));
  }

  return best_intersection;
}
} // namespace detail
} // namespace loop_all
} // namespace accel
} // namespace intersect
