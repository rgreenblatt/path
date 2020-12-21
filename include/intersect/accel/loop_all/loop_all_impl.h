#pragma once

#include "intersect/accel/add_idx.h"
#include "intersect/accel/loop_all/loop_all.h"

namespace intersect {
namespace accel {
namespace loop_all {
namespace detail {
template <Object O>
HOST_DEVICE inline AccelRet<O>
Ref::intersect_objects(const intersect::Ray &ray, Span<const O> objects) const {
  AccelRet<O> best;

  for (unsigned idx = 0; idx < size; idx++) {
    best = optional_min(best, add_idx(objects[idx].intersect(ray), idx));
  }

  return best;
}
} // namespace detail
} // namespace loop_all
} // namespace accel
} // namespace intersect
