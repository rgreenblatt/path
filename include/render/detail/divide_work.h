#pragma once

namespace render {
namespace detail {
// It is possible to have strategies which use n warps or n threads, but these
// aren't implemented right now.
enum class ReductionStrategy {
  Block,
  Warp,
  Thread,
};

struct WorkDivision {
  ReductionStrategy sample_reduction_strategy;
  unsigned block_size;
  unsigned samples_per_thread;
  unsigned x_block_size;
  unsigned y_block_size;
  unsigned num_sample_blocks; // Always 1 right now
  unsigned num_x_blocks;
  unsigned num_y_blocks;
};

WorkDivision divide_work(unsigned samples_per, unsigned x_dim, unsigned y_dim);
} // namespace detail
} // namespace render
