#pragma once

#include "intersect/accel/accel.h"
#include "intersect/accel/detail/node.h"
#include "lib/span.h"

namespace intersect {
namespace accel {
namespace detail {
struct BVH {
  SpanSized<const Node> nodes;

  // TODO: remove this hack...
  unsigned start_idx;

  template <IntersectableAtIdx F>
  HOST_DEVICE inline AccelRet<F>
  intersect_objects(const intersect::Ray &ray,
                    const F &intersectable_at_idx) const;
};

static_assert(AccelRef<BVH>);
} // namespace detail
} // namespace accel
} // namespace intersect
