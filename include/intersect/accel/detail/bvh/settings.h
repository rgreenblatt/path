#pragma once

#include "lib/settings.h"

namespace intersect {
namespace accel {
namespace detail {
namespace bvh {
struct Settings {
  // for SA heuristic
  // this assumes single intersection cost (of one)
  // TODO: this maybe should be lower, but that causes infinitely
  // deep SBVH hierarchies! This will be fixed more directly later...
  float traversal_per_intersect_cost = 0.5;

  unsigned target_objects = 4;

  bool print_stats = false;

  SETTING_BODY(Settings, traversal_per_intersect_cost, target_objects,
               print_stats);
};

static_assert(Setting<Settings>);
} // namespace bvh
} // namespace detail
} // namespace accel
} // namespace intersect
