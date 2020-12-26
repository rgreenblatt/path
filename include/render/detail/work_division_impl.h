#pragma once

#include "lib/cuda/reduce.cuh"
#include "lib/cuda/utils.h"
#include "render/detail/work_division.h"

#include <cassert>

namespace render {
namespace detail {
inline HOST_DEVICE WorkDivision::ThreadInfo
WorkDivision::get_thread_info(unsigned block_idx, unsigned thread_idx,
                              unsigned samples_per) const {
  const unsigned block_idx_sample = block_idx % num_sample_blocks_;
  const unsigned block_idx_pixel = block_idx / num_sample_blocks_;
  const unsigned block_idx_x = block_idx_pixel % num_x_blocks_;
  const unsigned block_idx_y = block_idx_pixel / num_x_blocks_;

  unsigned work_idx = samples_per_thread_ * thread_idx;

  unsigned sample_block_size = samples_per / num_sample_blocks_;

  const unsigned work_idx_sample = work_idx % sample_block_size;
  const unsigned work_idx_pixel = work_idx / sample_block_size;
  const unsigned work_idx_x = work_idx_pixel % x_block_size_;
  const unsigned work_idx_y = work_idx_pixel / x_block_size_;

  const unsigned start_sample =
      work_idx_sample + block_idx_sample * sample_block_size;
  const unsigned end_sample = start_sample + samples_per_thread_;
  const unsigned x = work_idx_x + block_idx_x * x_block_size_;
  const unsigned y = work_idx_y + block_idx_y * y_block_size_;

  return {start_sample, end_sample, x, y};
}

template <typename F>
DEVICE void WorkDivision::call_with_reduce(unsigned thread_idx,
                                           unsigned block_idx, F &&f) const {
  if (sample_reduction_strategy_ == ReductionStrategy::Block) {
    f(
        [&](const auto &v, auto &&oper, const auto &init) {
          return block_reduce(v, oper, init, thread_idx, block_size_);
        },
        thread_idx == 0, block_idx % num_sample_blocks_);
  } else if (sample_reduction_strategy_ == ReductionStrategy::Warp) {
    f([&](const auto &v, auto &&oper,
          const auto & /*init*/) { return warp_reduce(v, oper); },
      (thread_idx % warp_size) == 0, 0);
  } else if (sample_reduction_strategy_ == ReductionStrategy::Thread) {
    f([&](const float v, auto && /*op*/, const auto & /*init*/) { return v; },
      true, 0);
  }
}
} // namespace detail
} // namespace render
