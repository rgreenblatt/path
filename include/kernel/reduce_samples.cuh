#pragma once

#include "kernel/work_division.h"
#include "lib/cuda/reduce.cuh"

namespace kernel {
template <std::copyable T, typename BinOp>
inline __device__ T reduce_samples(const WorkDivision &division, const T &val,
                                   const BinOp &op, unsigned thread_idx) {
  return sub_block_reduce(val, op, thread_idx, division.block_size(),
                          division.sample_block_size());
}
} // namespace kernel
