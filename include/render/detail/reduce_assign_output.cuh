#pragma once

#include "lib/cuda/utils.h"
#include "meta/specialization_of.h"
#include "render/detail/assign_output.h"
#include "render/detail/integrate_image_base_items.h"
#include "work_division/reduce_samples.cuh"
#include "work_division/work_division.h"

#include <Eigen/Core>

namespace render {
namespace detail {
DEVICE void reduce_assign_output(const IntegrateImageBaseItems &b,
                                 const work_division::WorkDivision &division,
                                 unsigned thread_idx, unsigned block_idx,
                                 unsigned x, unsigned y,
                                 const Eigen::Array3f &intensity) {

  Eigen::Array3f totals;
  for (unsigned axis = 0; axis < 3; axis++) {
    auto add = [](auto lhs, auto rhs) { return lhs + rhs; };
    totals[axis] = reduce_samples(division, intensity[axis], add, thread_idx);
  }

  if (division.assign_sample(thread_idx)) {
    assign_output(b, division, division.sample_block_idx(block_idx),
                  division.num_sample_blocks(), x, y, totals);
  }
}
} // namespace detail
} // namespace render
