#pragma once

#include "execution_model/execution_model.h"
#include "kernel/detail/kernel_launch_impl.h"
#include "kernel/kernel_launch.h"
#include "kernel/work_division.h"
#include "lib/assert.h"
#include "lib/cuda/utils.h"
#include "meta/tuple.h"

namespace kernel {
template <Launchable L>
__global__ void gpu_kernel(const WorkDivision division, unsigned start_block,
                           L l) {
  const unsigned block_idx = blockIdx.x + start_block;
  const unsigned thread_idx = threadIdx.x;
  auto ref = l.block_init(division, block_idx);

  detail::kernel_launch_run(division, block_idx, thread_idx, ref);
}

template <>
template <Launchable L>
void KernelLaunch<ExecutionModel::GPU>::run_internal(
    const WorkDivision &division, unsigned start_block, unsigned end_block,
    const L &l, bool sync) {
  gpu_kernel<<<end_block - start_block, division.block_size()>>>(
      division, start_block, l);
  if (sync) {
    CUDA_SYNC_CHK();
  }
}
} // namespace kernel
