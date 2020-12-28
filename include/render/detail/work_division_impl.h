#pragma once

#include "lib/assert.h"
#include "lib/cuda/reduce.cuh"
#include "lib/cuda/utils.h"
#include "render/detail/work_division.h"

#include <algorithm>

namespace render {
namespace detail {
inline HOST_DEVICE WorkDivision::ThreadInfo
WorkDivision::get_thread_info(unsigned block_idx, unsigned thread_idx) const {
  debug_assert(thread_idx < block_size_);

  // handling block_idx and thread_idx separately like this ensures we don't
  // have overflow without having to use 64 bit integers (which are slow on
  // gpus)

  unsigned block_idx_sample = block_idx % num_sample_blocks_;
  unsigned block_idx_location = block_idx / num_sample_blocks_;
  unsigned block_idx_x = block_idx_location % num_x_blocks_;
  unsigned block_idx_y = block_idx_location / num_x_blocks_;

  unsigned thread_idx_sample = thread_idx % sample_block_size_;
  unsigned thread_idx_location = thread_idx / sample_block_size_;
  unsigned thread_idx_x = thread_idx_location % x_block_size_;
  unsigned thread_idx_y = thread_idx_location / x_block_size_;

  unsigned sample_idx =
      thread_idx_sample + block_idx_sample * sample_block_size_;
  unsigned x = thread_idx_x + block_idx_x * x_block_size_;
  unsigned y = thread_idx_y + block_idx_y * y_block_size_;

  unsigned base_samples_before = sample_idx * base_samples_per_thread_;
  unsigned n_extra_sample_before =
      std::min(n_threads_per_unit_extra_, sample_idx);
  unsigned start_sample = base_samples_before + n_extra_sample_before;
  bool has_extra_sample = sample_idx < n_threads_per_unit_extra_; 
  unsigned n_samples = base_samples_per_thread_ +
                       (has_extra_sample ? 1 : 0);

  unsigned end_sample = start_sample + n_samples;

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
