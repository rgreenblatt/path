#pragma once

#include "execution_model/execution_model.h"
#include "kernel/detail/kernel_launch_impl.h"
#include "kernel/kernel_launch.h"
#include "kernel/work_division.h"
#include "lib/assert.h"
#include "lib/cuda/utils.h"
#include "meta/tuple.h"

namespace kernel {
template <ThreadInteractor... Interactors, Launchable<Interactors...> F>
__global__ void gpu_kernel(const WorkDivision division, unsigned start_block,
                           F f) {
  const unsigned block_idx = blockIdx.x + start_block;
  const unsigned thread_idx = threadIdx.x;

  // interactors per thread
  MetaTuple<Interactors...> interactors_tup{Interactors(division)...};
  detail::kernel_launch_run(division, block_idx, thread_idx, f,
                            interactors_tup);
}

template <>
template <ThreadInteractor... Interactors, Launchable<Interactors...> F>
void KernelLaunch<ExecutionModel::GPU>::run_internal(
    const WorkDivision &division, unsigned start_block, unsigned end_block,
    const F &f, bool sync) {
  gpu_kernel<Interactors...>
      <<<end_block - start_block, division.block_size()>>>(division,
                                                           start_block, f);
  if (sync) {
    CUDA_SYNC_CHK();
  }
}
} // namespace kernel
