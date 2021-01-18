#pragma once

#include "execution_model/execution_model.h"
#include "kernel/detail/kernel_launch_impl.h"
#include "kernel/kernel_launch.h"
#include "kernel/work_division.h"
#include "lib/assert.h"

namespace kernel {
template <>
template <Launchable L>
void KernelLaunch<ExecutionModel::CPU>::run_internal(
    const WorkDivision &division, unsigned start_block, unsigned end_block,
    const L &launchable_in, bool /* sync */) {
  // TODO: better scheduling approach?  allow input argument to control?
#ifdef NDEBUG
#pragma omp parallel for schedule(dynamic, 1)
#endif
#if 0
  for (unsigned block_idx = start_block; block_idx < end_block; block_idx++) {
    L launchable = launchable_in;
    auto ref = launchable.block_init(division, block_idx);
    for (unsigned thread_idx = 0; thread_idx < division.block_size();
         thread_idx++) {
      detail::kernel_launch_run(division, block_idx, thread_idx, ref);
    }
  }
#endif
}
} // namespace kernel
