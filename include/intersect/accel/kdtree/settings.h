#pragma once

#include "intersect/accel/s_a_heuristic_settings.h"
#include "lib/settings.h"

namespace intersect {
namespace accel {
namespace kdtree {
struct Settings {
  SAHeuristicSettings s_a_heuristic = {1, 1};

  // use heuristic or just sort using centers of bounds
  bool use_s_a_heuritic = false; // not yet implemented

  // if there are less than or equal to than this many objects in a given node,
  // terminate
  unsigned num_objects_terminate = 2;

  template <typename Archive> void serialize(Archive &archive) {
    archive(NVP(s_a_heuristic), NVP(use_s_a_heuritic),
            NVP(num_objects_terminate));
  }

  constexpr bool operator==(const Settings &) const = default;
};

static_assert(Setting<Settings>);
} // namespace kdtree
} // namespace accel
} // namespace intersect
