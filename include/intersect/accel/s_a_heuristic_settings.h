#pragma once

#include "lib/settings.h"

namespace intersect {
namespace accel {
struct SAHeuristicSettings {
  float traversal_cost;
  float intersection_cost;

  SETTING_BODY(SAHeuristicSettings, traversal_cost, intersection_cost);
};

static_assert(Setting<SAHeuristicSettings>);
} // namespace accel
} // namespace intersect
