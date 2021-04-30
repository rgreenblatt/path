#pragma once

#include "lib/settings.h"

namespace intersect {
namespace accel {
namespace sbvh {
struct Settings {
  // NOTE: this assumes single intersection cost
  float traversal_per_intersect_cost = 2.;

  // alpha from original paper
  float overlap_threshold = 1e-5;

  SETTING_BODY(Settings, traversal_per_intersect_cost, overlap_threshold);
};

static_assert(Setting<Settings>);
} // namespace sbvh
} // namespace accel
} // namespace intersect
