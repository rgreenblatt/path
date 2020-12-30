#include "work_division/work_division.h"
#include "lib/assert.h"
#include "lib/bit_utils.h"
#include "lib/integer_division_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "lib/info/debug_print.h"

namespace work_division {
WorkDivision::WorkDivision(const Settings &settings, unsigned samples_per,
                           unsigned x_dim, unsigned y_dim) {
  block_size_ = settings.block_size;

  always_assert(block_size_ % warp_size == 0);
  static_assert(power_of_2(warp_size));
  always_assert(power_of_2(block_size_));
  always_assert(power_of_2(settings.target_x_block_size));

  unsigned n_threads_per_location = closest_power_of_2(
      static_cast<unsigned>(std::round(static_cast<float>(samples_per) /
                                       settings.target_samples_per_thread)));

  base_samples_per_thread_ = samples_per / n_threads_per_location;
  n_threads_per_unit_extra_ =
      samples_per - base_samples_per_thread_ * n_threads_per_location;

  sample_block_size_ = std::min(n_threads_per_location, block_size_);

  unsigned x_y_block_size = block_size_ / sample_block_size_;
  x_block_size_ = std::min(x_y_block_size, settings.target_x_block_size);
  y_block_size_ = x_y_block_size / x_block_size_;

  always_assert(power_of_2(sample_block_size_));
  always_assert(power_of_2(x_block_size_));
  always_assert(power_of_2(y_block_size_));

  num_sample_blocks_ = std::max(n_threads_per_location / block_size_, 1u);
  num_x_blocks_ = ceil_divide(x_dim, x_block_size_);
  num_y_blocks_ = ceil_divide(y_dim, y_block_size_);

  // dbg(base_samples_per_thread_);
  // dbg(n_threads_per_unit_extra_);

  always_assert(static_cast<uint64_t>(num_sample_blocks_) * num_x_blocks_ *
                    num_y_blocks_ <
                static_cast<uint64_t>(std::numeric_limits<unsigned>::max()));
}
} // namespace work_division
