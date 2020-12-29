#include "render/detail/work_division.h"
#include "lib/assert.h"
#include "lib/integer_division_utils.h"

#include <cmath>
#include <algorithm>
#include <cstdint>
#include <limits>

#include "lib/info/debug_print.h"

namespace render {
namespace detail {
WorkDivision::WorkDivision(const WorkDivisionSettings &settings,
                           unsigned samples_per, unsigned x_dim,
                           unsigned y_dim) {
  block_size_ = settings.block_size;
  x_block_size_ = 1;
  y_block_size_ = 1;
  num_sample_blocks_ = 1;

  always_assert(block_size_ ==
                settings.target_x_block_size * settings.target_y_block_size);
  always_assert(block_size_ % warp_size == 0);

  unsigned warps_per_block = settings.block_size / warp_size;

  unsigned n_threads_per_location;

  if (samples_per >= warp_size * settings.max_samples_per_thread) {
    // reduce over blocks
    unsigned target_samples_per_block =
        block_size_ * settings.target_samples_per_thread;
    num_sample_blocks_ = std::max(
        static_cast<unsigned>(std::round(static_cast<float>(samples_per) /
                                         target_samples_per_block)),
        1u);
    n_threads_per_location = num_sample_blocks_ * block_size_;
  } else if (samples_per >= settings.max_samples_per_thread) {
    // reduce over warp
    x_block_size_ = warps_per_block;
    n_threads_per_location = warp_size;
  } else {
    // reduce over single thread
    n_threads_per_location = 1;
    x_block_size_ = settings.target_x_block_size;
    y_block_size_ = settings.target_y_block_size;
  }

  base_samples_per_thread_ = samples_per / n_threads_per_location;
  n_threads_per_unit_extra_ =
      samples_per - base_samples_per_thread_ * n_threads_per_location;
  sample_block_size_ = std::min(n_threads_per_location, block_size_);

  num_x_blocks_ = ceil_divide(x_dim, x_block_size_);
  num_y_blocks_ = ceil_divide(y_dim, y_block_size_);

  // dbg(samples_per);
  // dbg(x_dim);
  // dbg(y_dim);
  // dbg(base_samples_per_thread_);
  // dbg(n_threads_per_unit_extra_);
  // dbg(sample_block_size_);
  // dbg(x_block_size_);
  // dbg(y_block_size_);
  // dbg(num_sample_blocks_);

  always_assert(static_cast<uint64_t>(num_sample_blocks_) * num_x_blocks_ *
                    num_y_blocks_ <
                static_cast<uint64_t>(std::numeric_limits<unsigned>::max()));
}
} // namespace detail
} // namespace render
