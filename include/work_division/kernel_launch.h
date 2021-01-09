#pragma once

#include "execution_model/execution_model.h"
#include "lib/assert.h"
#include "work_division/work_division.h"

namespace work_division {
// NOTE: Currently no mechanism to reduce on cpu...
template <ExecutionModel exec> struct KernelLaunch {
  template <typename F>
  static void run(const WorkDivision &work_division, unsigned start_block,
                  unsigned end_block, const F &f, bool sync = true) {
    debug_assert(end_block >= start_block);
    debug_assert(end_block <= work_division.total_num_blocks());

    run_internal(work_division, start_block, end_block, f, sync);
  }

private:
  template <typename F>
  static void run_internal(const WorkDivision &work_division,
                           unsigned start_block, unsigned end_block, const F &f,
                           bool sync);
};
} // namespace work_division
