#pragma once

#include "execution_model/execution_model.h"
#include "lib/assert.h"
#include "work_division/detail/kernel_launch_impl.h"
#include "work_division/kernel_launch.h"
#include "work_division/work_division.h"

namespace work_division {
template <>
template <typename F>
void KernelLaunch<ExecutionModel::CPU>::run_internal(
    const WorkDivision &division, unsigned start_block, unsigned end_block,
    const F &f, bool /* sync */) {
  // TODO: better scheduling approach?  allow input argument to control?
#ifdef NDEBUG
#pragma omp parallel for collapse(2) schedule(dynamic, 64)
#endif
  for (unsigned block_idx = start_block; block_idx < end_block; block_idx++) {
    for (unsigned thread_idx = 0; thread_idx < division.block_size();
         thread_idx++) {
      detail::kernel_launch_run(division, block_idx, thread_idx, f);
    }
  }
}
} // namespace work_division
