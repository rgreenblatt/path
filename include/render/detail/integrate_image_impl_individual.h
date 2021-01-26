#pragma once

#include "kernel/kernel_launch.h"
#include "kernel/make_runtime_constants_reduce_launchable.h"
#include "kernel/progress_bar_launch.h"
#include "lib/assert.h"
#include "lib/integer_division_utils.h"
#include "render/detail/integrate_image.h"
#include "render/detail/integrate_pixel.h"
#include "render/detail/max_blocks_per_launch.h"
#include "render/detail/reduce_assign_output.h"

namespace render {
namespace detail {
template <ExecutionModel exec>
template <typename... T>
void IntegrateImage<exec>::run(IntegrateImageIndividualInputs<T...> inp) {
  auto val = inp.val;

  kernel::progress_bar_launch(
      val.division,
      max_blocks_per_launch<exec>(inp.val.settings.computation_settings),
      val.show_progress, [&](unsigned start, unsigned end) {
        auto items = val.items;
        auto intersectable = val.intersector;
        auto settings = val.settings.rendering_equation_settings;

        kernel::KernelLaunch<exec>::run(
            val.division, start, end,
            kernel::make_runtime_constants_reduce_launchable<exec, FloatRGB>(
                [=](const WorkDivision &division,
                    const kernel::GridLocationInfo &info,
                    const unsigned block_idx, unsigned, const auto &,
                    auto &reducer) {
                  auto float_rgb = integrate_pixel(items, intersectable,
                                                   division, settings, info);

                  reduce_assign_output(reducer, items.base, division, block_idx,
                                       info.x, info.y, float_rgb);
                }));
      });
}
} // namespace detail
} // namespace render
