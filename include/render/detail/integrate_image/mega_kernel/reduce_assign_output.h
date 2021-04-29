#pragma once

#include "kernel/runtime_constants_reducer.h"
#include "kernel/work_division.h"
#include "lib/cuda/utils.h"
#include "lib/float_rgb.h"
#include "meta/specialization_of.h"
#include "render/detail/integrate_image/assign_output.h"
#include "render/detail/integrate_image/base_items.h"

namespace render {
namespace detail {
namespace integrate_image {
namespace mega_kernel {
template <typename F>
HOST_DEVICE void reduce_assign_output(F &reducer, const BaseItems &b,
                                      const kernel::WorkDivision &division,
                                      unsigned block_idx, unsigned x,
                                      unsigned y, const FloatRGB &float_rgb) {

  const std::optional<FloatRGB> total = reducer.reduce(
      float_rgb, [](const auto &lhs, const auto &rhs) { return lhs + rhs; },
      division.sample_block_size());

  if (total.has_value()) {
    assign_output(b, division.x_dim(), division.sample_block_idx(block_idx),
                  division.num_sample_blocks(), x, y, *total);
  }
}
} // namespace mega_kernel
} // namespace integrate_image
} // namespace detail
} // namespace render
