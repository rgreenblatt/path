#pragma once

#include "intersect/accel/accel.h"
#include "intersect/intersection.h"
#include "lib/optional.h"

namespace intersect {
namespace accel {
template <typename T>
HOST_DEVICE inline IntersectionOp<IdxHolder<T>>
add_idx(const IntersectionOp<T> &i, unsigned idx) {
  return optional_map(i, [&](const auto &v) {
    return v.map_info([&](const T &v) -> IdxHolder<T> { return {idx, v}; });
  });
}
} // namespace accel
} // namespace intersect
