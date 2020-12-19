#pragma once

#include "intersect/intersection.h"
#include "lib/optional.h"

namespace intersect {
namespace accel {
template <typename T>
HOST_DEVICE inline IntersectionOp<std::tuple<unsigned, T>>
add_idx(const IntersectionOp<T> &i, unsigned idx) {
  return optional_map(i, [&](const auto &v) {
    return v.map_info([&](const T &v) -> std::tuple<unsigned, T> {
      return {idx, v};
    });
  });
}
} // namespace accel
} // namespace intersect
