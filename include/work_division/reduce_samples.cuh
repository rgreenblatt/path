#pragma once

#include "lib/cuda/reduce.cuh"
#include "work_division/work_division.h"

namespace work_division {
template <std::copyable T, typename BinOp>
inline __device__ T reduce_samples(const WorkDivision &division, const T &val,
                                   const BinOp &op, unsigned thread_idx) {
  return sub_block_reduce(val, op, thread_idx, division.block_size(),
                          division.sample_block_size());
}
} // namespace work_division
