#pragma once

#include "lib/cuda/utils.h"
#include "lib/settings.h"

#include <compare>

namespace intersect {
namespace accel {
struct SAHeuristicSettings {
  float traversal_cost;
  float intersection_cost;

  HOST_DEVICE inline bool
  operator==(const SAHeuristicSettings &) const = default;
};

static_assert(Setting<SAHeuristicSettings>);
} // namespace accel
} // namespace intersect
