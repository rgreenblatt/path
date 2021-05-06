#pragma once

#include "intersect/accel/detail/bvh/settings.h"
#include "lib/settings.h"

namespace intersect {
namespace accel {
namespace direction_grid {
struct Settings {
  SETTING_BODY(Settings);
};

static_assert(Setting<Settings>);
} // namespace direction_grid
} // namespace accel
} // namespace intersect
