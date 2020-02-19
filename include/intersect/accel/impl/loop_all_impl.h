#pragma once

#include "intersect/accel/loop_all.h"

namespace intersect {
namespace accel {
template <ExecutionModel execution_model, typename Object>
HOST_DEVICE inline auto
LoopAll<execution_model, Object>::LoopAllRef::operator()(const Ray &ray) const {
  using PrevInfoType =
      typename std::decay_t<decltype(*std::declval<Object>()(Ray()))>::InfoType;
  using NewInfoType = AppendIndexInfoType<PrevInfoType>;
  using IntersectionOpT = IntersectionOp<NewInfoType>;

  IntersectionOpT best_intersection;
  for (unsigned idx = 0; idx < objects_.size(); idx++) {
    auto intersection = objects_[idx](ray);
    best_intersection = optional_min(best_intersection,
                                     append_index(intersection, idx + offset_));
  }

  return best_intersection;
}
} // namespace accel
} // namespace intersect
