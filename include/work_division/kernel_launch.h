#pragma once

#include "execution_model/execution_model.h"
#include "lib/assert.h"
#include "lib/cuda/utils.h"
#include "work_division/launchable.h"
#include "work_division/thread_interactor.h"
#include "work_division/work_division.h"

namespace work_division {

// NOTE: Currently no mechanism to reduce on cpu...
// TODO: can I be cleverer here to avoid state and minimize math required for
// indexing???

// kernel launch - optimized for gpu, but it does support the cpu for easier
// development
template <ExecutionModel exec> struct KernelLaunch {
  // F can be a mutable lambda - it will be copied for each thread invokation
  template <ThreadInteractor... Interactors, Launchable<Interactors...> F>
  static void run(const WorkDivision &work_division, unsigned start_block,
                  unsigned end_block, const F &f, bool sync = true) {
    debug_assert(end_block >= start_block);
    debug_assert(end_block <= work_division.total_num_blocks());

    run_internal(work_division, start_block, end_block, f, sync);
  }

private:
  template <ThreadInteractor... Interactors, Launchable<Interactors...> F>
  static void run_internal(const WorkDivision &work_division,
                           unsigned start_block, unsigned end_block, const F &f,
                           bool sync);
};
} // namespace work_division
