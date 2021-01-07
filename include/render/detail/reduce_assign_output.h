#pragma once

#include "lib/cuda/utils.h"
#include "meta/specialization_of.h"
#include "render/detail/assign_output.h"
#include "render/detail/integrate_image_base_items.h"
#include "work_division/work_division_impl.h"

#include <Eigen/Core>

namespace render {
namespace detail {
DEVICE void reduce_assign_output(unsigned thread_idx, unsigned block_idx,
                                 const IntegrateImageBaseItems &b, unsigned x,
                                 unsigned y, const Eigen::Array3f &intensity) {

  Eigen::Array3f totals;
  for (unsigned axis = 0; axis < 3; axis++) {
    auto add = [](auto lhs, auto rhs) { return lhs + rhs; };
    totals[axis] = b.division.reduce_samples(intensity[axis], add, thread_idx);
  }

  if (b.division.assign_sample(thread_idx)) {
    assign_output(b, b.division.sample_block_idx(block_idx),
                  b.division.num_sample_blocks(), x, y, totals);
  }
}
} // namespace detail
} // namespace render
