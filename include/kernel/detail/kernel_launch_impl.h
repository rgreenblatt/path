#pragma once

#include "kernel/launchable.h"
#include "kernel/work_division.h"
#include "kernel/work_division_impl.h"
#include "lib/cuda/utils.h"

namespace kernel {
namespace detail {
template <LaunchableBlockRef L>
HOST_DEVICE void kernel_launch_run(const WorkDivision &division,
                                   const unsigned block_idx,
                                   const unsigned thread_idx, L &l) {
  auto info = division.get_thread_info(block_idx, thread_idx);

  if (info.exit) {
    return;
  }

  l(division, info.info, block_idx, thread_idx);
}
} // namespace detail
} // namespace kernel
