#pragma once

#include "kernel/kernel_launch.h"
#include "kernel/thread_interactor_launchable.h"
#include "kernel/tuple_thread_interactor.h"
#include "kernel/work_division.h"
#include "lib/assert.h"
#include "render/detail/integrate_image_base_items.h"
#include "render/detail/reduce_float_rgb.h"
// #include "render/detail/reduce_assign_output.cuh"
// #include "render/detail/reduce_float_rgb.h"
#pragma message "more includes"

#include "data_structure/copyable_to_vec.h"

namespace render {
namespace detail {
template <ExecutionModel exec>
ExecVector<exec, FloatRGB> *ReduceFloatRGB<exec>::run(
    bool output_as_bgra_32, unsigned reduction_factor, unsigned samples_per,
    ExecVector<exec, FloatRGB> *float_rgb_in,
    ExecVector<exec, FloatRGB> *float_rgb_out, Span<BGRA32> bgras) {
  while (reduction_factor != 1) {
    always_assert(float_rgb_in->size() % reduction_factor == 0);
    unsigned x_dim = float_rgb_in->size() / reduction_factor;
    // TODO: SPEED: reduce division settings
    kernel::WorkDivision division({.block_size = 256,
                                   .target_x_block_size = 256,
                                   .force_target_samples = false,
                                   .forced_target_samples_per_thread = 1,
                                   .base_num_threads = 16384,
                                   .samples_per_thread_scaling_power = 0.5f,
                                   .max_samples_per_thread = 8},
                                  reduction_factor, x_dim, 1);
    float_rgb_out->resize(x_dim * division.num_sample_blocks());

    IntegrateImageBaseItems items{.output_as_bgra_32 =
                                      output_as_bgra_32 &&
                                      division.num_sample_blocks() == 1,
                                  .samples_per = samples_per,
                                  .bgra_32 = bgras,
                                  .float_rgb = *float_rgb_out};

    Span<const FloatRGB> in_span = *float_rgb_in;

    using ExtraInp = kernel::EmptyExtraInp;

    auto callable = [=](const kernel::WorkDivision &division,
                        const kernel::GridLocationInfo &info,
                        const unsigned block_idx, const unsigned thread_idx,
                        ExtraInp, auto) {
      auto [start_sample, end_sample, x, y] = info;

      FloatRGB total = FloatRGB::Zero();
      for (unsigned i = start_sample; i < end_sample; ++i) {
        total += in_span[i + x * reduction_factor];
      }

#pragma message "REDUCE"
      // reduce_assign_output(items, division, thread_idx,
      // block_idx,
      //                      x, 0, total);
    };

    kernel::KernelLaunch<exec>::run(
        division, 0, division.total_num_blocks(),
        kernel::ThreadInteractorLaunchable<
            ExtraInp, kernel::TupleThreadInteractor<ExtraInp>,
            decltype(callable)>{
            .inp = {},
            .interactor = {},
            .callable = callable,
        });

    reduction_factor = division.num_sample_blocks();

    std::swap(float_rgb_in, float_rgb_out);
  }

  return float_rgb_in;
}
} // namespace detail
} // namespace render
