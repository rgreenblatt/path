#pragma once

#include "lib/cuda//utils.h"

#include <compare>

namespace intersect {
namespace accel {
struct SAHeuristicSettings {
  float traversal_cost;
  float intersection_cost;

  HOST_DEVICE inline bool
  operator==(const SAHeuristicSettings &) const = default;
};
} // namespace accel
} // namespace intersect
