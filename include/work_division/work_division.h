#pragma once

#include "lib/attribute.h"
#include "lib/cuda/utils.h"
#include "lib/settings.h"
#include "work_division/settings.h"

namespace work_division {
class WorkDivision {
public:
  WorkDivision() {}

  WorkDivision(const Settings &settings, unsigned samples_per, unsigned x_dim,
               unsigned y_dim);

  struct ThreadInfo {
    unsigned start_sample;
    unsigned end_sample;
    unsigned x;
    unsigned y;
  };

  ATTR_PURE_NDEBUG inline HOST_DEVICE ThreadInfo
  get_thread_info(unsigned block_idx, unsigned thread_idx) const;

  template <typename T, typename BinOp>
  inline DEVICE T reduce_samples(const T &val, const BinOp &op,
                                 unsigned thread_idx) const;

  ATTR_PURE inline HOST_DEVICE bool assign_sample(unsigned thread_idx) const {
    return thread_idx % sample_block_size_ == 0;
  }

  ATTR_PURE inline HOST_DEVICE unsigned
  sample_block_idx(unsigned block_idx) const {
    return block_idx % num_sample_blocks_;
  }

  ATTR_PURE HOST_DEVICE unsigned block_size() const { return block_size_; }
  ATTR_PURE HOST_DEVICE unsigned base_samples_per_thread() const {
    return base_samples_per_thread_;
  }
  ATTR_PURE HOST_DEVICE unsigned n_threads_per_unit_extra() const {
    return n_threads_per_unit_extra_;
  }
  ATTR_PURE HOST_DEVICE unsigned sample_block_size() const {
    return sample_block_size_;
  }
  ATTR_PURE HOST_DEVICE unsigned x_block_size() const { return x_block_size_; }
  ATTR_PURE HOST_DEVICE unsigned y_block_size() const { return y_block_size_; }
  ATTR_PURE HOST_DEVICE unsigned num_sample_blocks() const {
    return num_sample_blocks_;
  }
  ATTR_PURE HOST_DEVICE unsigned num_x_blocks() const { return num_x_blocks_; }
  ATTR_PURE HOST_DEVICE unsigned num_y_blocks() const { return num_y_blocks_; }

  ATTR_PURE HOST_DEVICE unsigned total_num_blocks() const {
    return num_sample_blocks_ * num_x_blocks_ * num_y_blocks_;
  }

private:
  unsigned block_size_;
  unsigned base_samples_per_thread_;
  unsigned n_threads_per_unit_extra_;
  unsigned sample_block_size_;
  unsigned x_block_size_;
  unsigned y_block_size_;
  unsigned num_sample_blocks_;
  unsigned num_x_blocks_;
  unsigned num_y_blocks_;
};
} // namespace work_division
