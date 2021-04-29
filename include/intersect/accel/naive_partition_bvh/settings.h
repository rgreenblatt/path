#pragma once

#include "intersect/accel/s_a_heuristic_settings.h"
#include "lib/settings.h"

namespace intersect {
namespace accel {
namespace naive_partition_bvh {
struct Settings {
  // if there are less than or equal to than this many objects in a given node,
  // terminate
  unsigned num_objects_terminate = 2;

  SETTING_BODY(Settings, num_objects_terminate);
};

static_assert(Setting<Settings>);
} // namespace naive_partition_bvh
} // namespace accel
} // namespace intersect
