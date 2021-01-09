#pragma once

#include "execution_model/execution_model.h"
#include "lib/assert.h"
#include "lib/cuda/utils.h"
#include "work_division/detail/kernel_launch_impl.h"
#include "work_division/kernel_launch.h"
#include "work_division/work_division.h"

namespace work_division {
template <typename F>
__global__ void gpu_kernel(const WorkDivision division, unsigned start_block,
                           const F f) {
  const unsigned block_idx = blockIdx.x + start_block;
  const unsigned thread_idx = threadIdx.x;

  detail::kernel_launch_run(division, block_idx, thread_idx, f);
}

template <>
template <typename F>
void KernelLaunch<ExecutionModel::GPU>::run_internal(
    const WorkDivision &division, unsigned start_block, unsigned end_block,
    const F &f, bool sync) {
  gpu_kernel<<<end_block - start_block, division.block_size()>>>(
      division, start_block, f);
  if (sync) {
    CUDA_ERROR_CHK(cudaDeviceSynchronize());
    CUDA_ERROR_CHK(cudaGetLastError());
  }
}
} // namespace work_division
