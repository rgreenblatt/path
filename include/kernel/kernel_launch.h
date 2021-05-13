#pragma once

#include "execution_model/execution_model.h"
#include "execution_model/thrust_data.h"
#include "kernel/launchable.h"
#include "kernel/work_division.h"
#include "lib/assert.h"
#include "lib/cuda/utils.h"

namespace kernel {
// TODO: can I be cleverer here (with dispatching) to avoid state and minimize
// math required for indexing???

// kernel launch - optimized for gpu, but it does support the cpu for easier
// development/debugging
template <ExecutionModel exec> struct KernelLaunch {
  // F can be a mutable lambda - it will be copied for each thread invokation
  template <Launchable L>
  static void run(const ThrustData<exec> &data, const WorkDivision &kernel,
                  unsigned start_block, unsigned end_block, const L &launchable,
                  bool sync = true) {
    debug_assert(end_block >= start_block);
    debug_assert(end_block <= kernel.total_num_blocks());

    run_internal(data, kernel, start_block, end_block, launchable, sync);
  }

private:
  template <Launchable L>
  static void run_internal(const ThrustData<exec> &data,
                           const WorkDivision &kernel, unsigned start_block,
                           unsigned end_block, const L &launchable, bool sync);
};
} // namespace kernel
