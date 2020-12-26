#include "render/detail/work_division.h"
#include "lib/cuda/utils.h"
#include "lib/integer_division_utils.h"

#include <cassert>

namespace render {
namespace detail {
WorkDivision::WorkDivision(
    const WorkDivisionSettings &settings, // TODO settings
    unsigned &samples_per, unsigned x_dim, unsigned y_dim) {
  assert(settings.block_size ==
         settings.target_x_block_size * settings.target_y_block_size);
  assert(settings.block_size % warp_size == 0);

  const unsigned warps_per_block = settings.block_size / warp_size;

  unsigned num_sample_blocks = 1;

  ReductionStrategy sample_reduction_strategy;
  unsigned samples_per_thread;
  unsigned x_block_size = 1;
  unsigned y_block_size = 1;

  if (samples_per >= warp_size * settings.max_samples_per_thread) {
    sample_reduction_strategy = ReductionStrategy::Block;
    unsigned samples_per_block =
        settings.block_size * settings.target_samples_per_thread;
    samples_per = make_divisable(samples_per, samples_per_block);
    samples_per_thread = settings.target_samples_per_thread;
    num_sample_blocks = samples_per / samples_per_block;
  } else if (samples_per >= settings.max_samples_per_thread) {
    sample_reduction_strategy = ReductionStrategy::Warp;
    samples_per = make_divisable(samples_per, warp_size);
    samples_per_thread = samples_per / warp_size;
    x_block_size = warps_per_block;
  } else {
    sample_reduction_strategy = ReductionStrategy::Thread;
    samples_per_thread = samples_per;
    x_block_size = settings.target_x_block_size;
    y_block_size = settings.target_y_block_size;
  }

  unsigned num_x_blocks = ceil_divide(x_dim, x_block_size);
  unsigned num_y_blocks = ceil_divide(y_dim, y_block_size);

  sample_reduction_strategy_ = sample_reduction_strategy;
  block_size_ = settings.block_size;
  samples_per_thread_ = samples_per_thread;
  x_block_size_ = x_block_size;
  y_block_size_ = y_block_size;
  num_sample_blocks_ = num_sample_blocks;
  num_x_blocks_ = num_x_blocks;
  num_y_blocks_ = num_y_blocks;
}
} // namespace detail
} // namespace render
