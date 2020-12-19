#pragma once

#include "intersect/accel/loop_all.h"

namespace intersect {
template <accel::LoopAllRef Ref> struct IntersectableImpl<Ref> {
  template <typename... T>
  HOST_DEVICE static inline auto intersect(const Ray &ray, const Ref &ref,
                                           T... vals) {
    using O = typename Ref::InstO;
    using IntersectableO = IntersectableT<O>;
    using PrevInfoType = typename IntersectableO::template Intersection<T...>;
    using NewInfoType = AppendIndexInfoType<PrevInfoType>;
    using IntersectionOpT = IntersectionOp<NewInfoType>;

    IntersectionOpT best_intersection;

    for (unsigned idx = 0; idx < ref.objects_.size(); idx++) {
      auto intersection =
          IntersectableO::intersect(ray, ref.objects_[idx], vals...);
      best_intersection = optional_min(
          best_intersection, append_index(intersection, idx + ref.offset_));
    }

    return best_intersection;
  }
};

template <accel::LoopAllRef Ref> struct BoundedImpl<Ref> {
  static HOST_DEVICE inline const accel::AABB &bounds(const Ref &ref) {
    return ref.aabb();
  }
};
} // namespace intersect
