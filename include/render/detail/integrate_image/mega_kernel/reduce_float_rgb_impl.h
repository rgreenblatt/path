#pragma once

#include "kernel/kernel_launch.h"
#include "kernel/make_runtime_constants_reduce_launchable.h"
#include "kernel/work_division.h"
#include "lib/assert.h"
#include "render/detail/integrate_image/base_items.h"
#include "render/detail/integrate_image/mega_kernel/reduce_assign_output.h"
#include "render/detail/integrate_image/mega_kernel/reduce_float_rgb.h"

#include "data_structure/copyable_to_vec.h"

namespace render {
namespace detail {
namespace integrate_image {
namespace mega_kernel {
template <ExecutionModel exec>
ExecVector<exec, FloatRGB> *ReduceFloatRGB<exec>::run(
    const kernel::WorkDivisionSettings &division_settings,
    bool output_as_bgra_32, unsigned reduction_factor, unsigned samples_per,
    ExecVector<exec, FloatRGB> *float_rgb_in,
    ExecVector<exec, FloatRGB> *float_rgb_out, Span<BGRA32> bgras) {
  while (reduction_factor != 1) {
    always_assert(float_rgb_in->size() % reduction_factor == 0);
    unsigned x_dim = float_rgb_in->size() / reduction_factor;
    kernel::WorkDivision division(division_settings, reduction_factor, x_dim,
                                  1);
    float_rgb_out->resize(x_dim * division.num_sample_blocks());

    BaseItems items{.output_as_bgra_32 =
                        output_as_bgra_32 && division.num_sample_blocks() == 1,
                    .samples_per = samples_per,
                    .bgra_32 = bgras,
                    .float_rgb = *float_rgb_out};

    Span<const FloatRGB> in_span = *float_rgb_in;

    kernel::KernelLaunch<exec>::run(
        division, 0, division.total_num_blocks(),
        kernel::make_runtime_constants_reduce_launchable<exec, FloatRGB>(
            [=](const kernel::WorkDivision &division,
                const kernel::GridLocationInfo &info, const unsigned block_idx,
                const unsigned, const auto &, auto &reducer) {
              auto [start_sample, end_sample, x, y] = info;

              FloatRGB total = FloatRGB::Zero();
              for (unsigned i = start_sample; i < end_sample; ++i) {
                total += in_span[i + x * reduction_factor];
              }

              reduce_assign_output(reducer, items, division, block_idx, info.x,
                                   info.y, total);
            }));

    reduction_factor = division.num_sample_blocks();

    std::swap(float_rgb_in, float_rgb_out);
  }

  return float_rgb_in;
}
} // namespace mega_kernel
} // namespace integrate_image
} // namespace detail
} // namespace render
