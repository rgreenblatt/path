#pragma once

#include "kernel/kernel_launch.h"
#include "kernel/make_runtime_constants_reduce_launchable.h"
#include "kernel/progress_bar_launch.h"
#include "lib/assert.h"
#include "lib/integer_division_utils.h"
#include "render/detail/integrate_image/items.h"
#include "render/detail/integrate_image/mega_kernel/integrate_pixel.h"
#include "render/detail/integrate_image/mega_kernel/reduce_assign_output.h"
#include "render/detail/integrate_image/mega_kernel/reduce_float_rgb.h"
#include "render/detail/integrate_image/mega_kernel/run.h"
#include "render/mega_kernel_settings.h"

namespace render {
namespace detail {
namespace integrate_image {
namespace mega_kernel {
namespace detail {
template <ExecutionModel exec>
unsigned
max_blocks_per_launch(const MegaKernelSettings::ComputationSettings &settings) {
  if constexpr (exec == ExecutionModel::GPU) {
    return settings.max_blocks_per_launch_gpu;
  } else {
    if (debug_build) {
      return 1;
    } else {
      return settings.max_blocks_per_launch_cpu;
    }
  }
}
} // namespace detail

template <ExecutionModel exec>
template <ExactSpecializationOf<Items> Items, intersect::Intersectable I>
requires std::same_as<typename Items::InfoType, typename I::InfoType>
    ExecVector<exec, FloatRGB>
*Run<exec>::run(Inputs<Items> inp, const I &intersectable,
                const MegaKernelSettings &settings,
                ExecVector<exec, FloatRGB> &float_rgb,
                ExecVector<exec, FloatRGB> &reduced_float_rgb) {
  kernel::WorkDivision division = {
      settings.computation_settings.render_work_division,
      inp.samples_per,
      inp.x_dim,
      inp.y_dim,
  };

  if (division.num_sample_blocks() != 1 || !inp.output_as_bgra_32) {
    float_rgb.resize(division.num_sample_blocks() * inp.x_dim * inp.y_dim);
  }

  auto base = BaseItems{
      .output_as_bgra_32 = inp.output_as_bgra_32,
      .samples_per = inp.samples_per,
      .bgra_32 = inp.bgra_32,
      .float_rgb = float_rgb,
  };

  kernel::progress_bar_launch(
      division,
      detail::max_blocks_per_launch<exec>(settings.computation_settings),
      inp.show_progress, [&](unsigned start, unsigned end) {
        auto items = inp.items;

        kernel::KernelLaunch<exec>::run(
            division, start, end,
            kernel::make_runtime_constants_reduce_launchable<exec, FloatRGB>(
                [=](const kernel::WorkDivision &division,
                    const kernel::GridLocationInfo &info,
                    const unsigned block_idx, unsigned, const auto &,
                    auto &reducer) {
                  auto float_rgb =
                      integrate_pixel(items, intersectable, division, info);

                  reduce_assign_output(reducer, base, division, block_idx,
                                       info.x, info.y, float_rgb);
                }));
      });

  auto float_rgb_reduce_out = ReduceFloatRGB<exec>::run(
      settings.computation_settings.reduce_work_division, inp.output_as_bgra_32,
      division.num_sample_blocks(), inp.samples_per, &float_rgb,
      &reduced_float_rgb, inp.bgra_32);
  always_assert(float_rgb_reduce_out != nullptr);
  always_assert(float_rgb_reduce_out == &float_rgb ||
                float_rgb_reduce_out == &reduced_float_rgb);

  return float_rgb_reduce_out;
}
} // namespace mega_kernel
} // namespace integrate_image
} // namespace detail
} // namespace render
