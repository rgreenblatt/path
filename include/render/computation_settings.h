#pragma once

#include "kernel/work_division_settings.h"
#include "lib/settings.h"

namespace render {
struct ComputationSettings {
  unsigned max_blocks_per_launch_gpu = 4096;
  // enough so threads are saturated, but doesn't need to be too much
  unsigned max_blocks_per_launch_cpu = 128;
  kernel::WorkDivisionSettings render_work_division = {};
  // TODO: SPEED: tune?
  kernel::WorkDivisionSettings reduce_work_division = {
      .block_size = 256, .target_x_block_size = 256};

  SETTING_BODY(ComputationSettings, max_blocks_per_launch_gpu,
               max_blocks_per_launch_cpu, render_work_division,
               reduce_work_division);
};

static_assert(Setting<ComputationSettings>);
} // namespace render
