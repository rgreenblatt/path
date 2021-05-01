#pragma once

#include "intersect/accel/detail/bvh/settings.h"
#include "lib/settings.h"

namespace intersect {
namespace accel {
namespace naive_partition_bvh {
struct Settings {
  // if there are less than or equal to than this many objects in a given node,
  // terminate
  unsigned num_objects_terminate = 1;
  detail::bvh::Settings bvh_settings;

  SETTING_BODY(Settings, num_objects_terminate, bvh_settings);
};

static_assert(Setting<Settings>);
} // namespace naive_partition_bvh
} // namespace accel
} // namespace intersect
