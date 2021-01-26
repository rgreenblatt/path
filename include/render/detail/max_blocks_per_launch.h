#pragma once

#include "execution_model/execution_model.h"
#include "lib/assert.h"
#include "render/computation_settings.h"

namespace render {
namespace detail {
template <ExecutionModel exec>
unsigned max_blocks_per_launch(const ComputationSettings &settings) {
  if constexpr (exec == ExecutionModel::GPU) {
    return settings.max_blocks_per_launch_gpu;
  } else {
    if (debug_build) {
      return 1;
    } else {
      return settings.max_blocks_per_launch_cpu;
    }
  }
}
} // namespace detail
} // namespace render
