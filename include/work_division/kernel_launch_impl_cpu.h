#pragma once

#include "execution_model/execution_model.h"
#include "lib/assert.h"
#include "meta/tuple.h"
#include "work_division/detail/kernel_launch_impl.h"
#include "work_division/kernel_launch.h"
#include "work_division/work_division.h"

namespace work_division {
template <>
template <ThreadInteractor... Interactors, Launchable<Interactors...> F>
void KernelLaunch<ExecutionModel::GPU>::run_internal(
    const WorkDivision &division, unsigned start_block, unsigned end_block,
    const F &f_in, bool /* sync */) {
  // TODO: better scheduling approach?  allow input argument to control?
#ifdef NDEBUG
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (unsigned block_idx = start_block; block_idx < end_block; block_idx++) {
    MetaTuple<Interactors...> interactors_tup{Interactors(division)...};
    for (unsigned thread_idx = 0; thread_idx < division.block_size();
         thread_idx++) {
      // state is individual to each thread, so we copy here
      F f = f_in;
      detail::kernel_launch_run(division, block_idx, thread_idx, f,
                                interactors_tup);
    }
  }
}
} // namespace work_division
