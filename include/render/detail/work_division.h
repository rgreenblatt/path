#pragma once

#include "lib/cuda/utils.h"
#include "lib/settings.h"
#include "render/work_division_settings.h"

namespace render {
namespace detail {
enum class ReductionStrategy {
  Block,
  Warp,
  Thread,
};

class WorkDivision {
public:
  WorkDivision() {}

  WorkDivision(const WorkDivisionSettings &settings, unsigned &samples_per,
               unsigned x_dim, unsigned y_dim);

  struct ThreadInfo {
    unsigned start_sample;
    unsigned end_sample;
    unsigned x;
    unsigned y;
  };

  inline HOST_DEVICE ThreadInfo get_thread_info(unsigned block_idx,
                                                unsigned thread_idx,
                                                unsigned samples_per) const;

  template <typename F>
  DEVICE void call_with_reduce(unsigned thread_idx, unsigned block_idx,
                               F &&f) const;

  HOST_DEVICE ReductionStrategy sample_reduction_strategy() const {
    return sample_reduction_strategy_;
  }
  HOST_DEVICE unsigned block_size() const { return block_size_; }
  HOST_DEVICE unsigned samples_per_thread() const {
    return samples_per_thread_;
  }
  HOST_DEVICE unsigned x_block_size() const { return x_block_size_; }
  HOST_DEVICE unsigned y_block_size() const { return y_block_size_; }
  HOST_DEVICE unsigned num_sample_blocks() const { return num_sample_blocks_; }
  HOST_DEVICE unsigned num_x_blocks() const { return num_x_blocks_; }
  HOST_DEVICE unsigned num_y_blocks() const { return num_y_blocks_; }

  HOST_DEVICE unsigned total_num_blocks() const {
    return num_sample_blocks_ * num_x_blocks_ * num_y_blocks_;
  }

private:
  ReductionStrategy sample_reduction_strategy_;
  unsigned block_size_;
  unsigned samples_per_thread_;
  unsigned x_block_size_;
  unsigned y_block_size_;
  unsigned num_sample_blocks_;
  unsigned num_x_blocks_;
  unsigned num_y_blocks_;
};
} // namespace detail
} // namespace render
