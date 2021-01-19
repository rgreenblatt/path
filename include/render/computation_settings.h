#pragma once

#include "kernel/settings.h"
#include "lib/settings.h"

namespace render {
struct ComputationSettings {
  unsigned max_blocks_per_launch = 4096;
  kernel::Settings render_work_division = {}; // TODO

  SETTING_BODY(ComputationSettings, max_blocks_per_launch,
               render_work_division);
};

static_assert(Setting<ComputationSettings>);
} // namespace render
