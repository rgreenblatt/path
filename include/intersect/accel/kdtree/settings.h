#pragma once

#include "intersect/accel/s_a_heuristic_settings.h"

namespace intersect {
namespace accel {
namespace kdtree {
struct Settings {
  SAHeuristicSettings s_a_heuristic;

  // use heuristic or just sort using centers of bounds
  bool use_s_a_heuritic;

  // if there are fewer than this many objects in a given node, terminate
  unsigned num_objects_terminate;

  HOST_DEVICE inline bool operator==(const Settings &) const = default;
};

static_assert(Setting<Settings>);
} // namespace kdtree
} // namespace accel
} // namespace intersect
