#pragma once

#include "intersect/accel/enum_accel/settings.h"
#include "kernel/work_division_settings.h"
#include "lib/settings.h"
#include "lib/tagged_union.h"
#include "render/individually_intersectable_settings.h"

namespace render {
struct MegaKernelSettings {
  struct ComputationSettings {
    unsigned max_blocks_per_launch_gpu = 4096;
    // enough so threads are saturated, but doesn't need to be too much
    unsigned max_blocks_per_launch_cpu = 128;
    kernel::WorkDivisionSettings render_work_division = {};
    // TODO: SPEED: tune?
    kernel::WorkDivisionSettings reduce_work_division = {
        .block_size = 256,
        .target_x_block_size = 256,
    };

    SETTING_BODY(ComputationSettings, max_blocks_per_launch_gpu,
                 max_blocks_per_launch_cpu, render_work_division,
                 reduce_work_division);
  };

  ComputationSettings computation_settings;
  IndividuallyIntersectableSettings individually_intersectable_settings;

  using CompileTime = IndividuallyIntersectableSettings::CompileTime;

  constexpr CompileTime compile_time() const {
    return individually_intersectable_settings.compile_time();
  }

  SETTING_BODY(MegaKernelSettings, computation_settings,
               individually_intersectable_settings);
};

static_assert(Setting<MegaKernelSettings::ComputationSettings>);
static_assert(Setting<MegaKernelSettings>);
} // namespace render
