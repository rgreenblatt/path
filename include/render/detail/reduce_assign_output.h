#pragma once

#include "lib/bgra.h"
#include "lib/cuda/utils.h"
#include "render/detail/assign_output.h"
#include "render/detail/work_division.h"
#include "render/detail/work_division_impl.h"

#include <Eigen/Core>

namespace render {
namespace detail {
DEVICE void
reduce_assign_output(unsigned thread_idx, unsigned block_idx,
                     bool output_as_bgra, unsigned x, unsigned y,
                     unsigned x_dim, const Eigen::Array3f &intensity,
                     Span<BGRA> bgras, Span<Eigen::Array3f> intensities,
                     const WorkDivision &division, unsigned samples_per) {
  division.call_with_reduce(
      thread_idx, block_idx,
      [&](const auto &reduce_func, bool assign, unsigned sample_block_idx) {
        Eigen::Array3f totals;
        for (unsigned axis = 0; axis < 3; axis++) {
          auto add = [](auto lhs, auto rhs) { return lhs + rhs; };
          totals[axis] = reduce_func(intensity[axis], add, 0.f);
        }
        if (assign) {
          assign_output(output_as_bgra, bgras, intensities, sample_block_idx,
                        division.num_sample_blocks(), x, y, x_dim, samples_per,
                        totals);
        }
      });
}
} // namespace detail
} // namespace render
