#pragma once

#include "lib/assert.h"
#include "lib/cuda/reduce.cuh"
#include "lib/cuda/utils.h"
#include "work_division/work_division.h"

#include <algorithm>

namespace work_division {
inline HOST_DEVICE WorkDivision::ThreadInfo
WorkDivision::get_thread_info(unsigned block_idx, unsigned thread_idx) const {
  debug_assert_assume(thread_idx < block_size_);

  // allow for compiler optimizations at the call site
  debug_assert_assume(block_size_ % warp_size == 0);
  debug_assert_assume(block_size_ >= warp_size);
  debug_assert_assume(block_size_ % sample_block_size_ == 0);
  debug_assert_assume(block_size_ % x_block_size_ == 0);
  debug_assert_assume(block_size_ % y_block_size_ == 0);
  debug_assert_assume(block_size_ ==
                      x_block_size_ * y_block_size_ * sample_block_size_);

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
  unsigned n_samples = base_samples_per_thread_ + (has_extra_sample ? 1 : 0);

  unsigned end_sample = start_sample + n_samples;

  return {start_sample, end_sample, x, y};
}

template <typename T, typename BinOp>
inline DEVICE T WorkDivision::reduce_samples(const T &val, const BinOp &op,
                                             unsigned thread_idx) const {
  return sub_block_reduce(val, op, thread_idx, block_size_, sample_block_size_);
}
} // namespace detail
