#pragma once

#include "kernel/kernel_launch_impl_gpu.cuh"
#include "lib/assert.h"
#include "lib/integer_division_utils.h"
#include "render/detail/integrate_image.h"
#include "render/detail/integrate_pixel.h"
#include "render/detail/reduce_assign_output.cuh"

#include <cli/ProgressBar.hpp>

namespace render {
namespace detail {
template <>
template <typename... T>
void IntegrateImage<ExecutionModel::GPU>::run(
    IntegrateImageIndividualInputs<T...> inp) {
  auto val = inp.val;
  unsigned total_grid = val.division.total_num_blocks();

  unsigned max_launch_size =
      val.settings.computation_settings.max_blocks_per_launch;

  unsigned num_launches = ceil_divide(total_grid, max_launch_size);
  unsigned blocks_per = total_grid / num_launches;

  ProgressBar progress_bar(num_launches, 70);
  if (val.show_progress) {
    progress_bar.display();
  }

  for (unsigned i = 0; i < num_launches; i++) {
    unsigned start = i * blocks_per;
    unsigned end = std::min((i + 1) * blocks_per, total_grid);

    auto items = val.items;
    auto intersectable = val.intersector;
    auto settings = val.settings.rendering_equation_settings;

#pragma message "fix this - kernel launch"
#if 0
    kernel::KernelLaunch<ExecutionModel::GPU>::run(
        val.division, start, end,
        [=](const WorkDivision &division, const kernel::GridLocationInfo &info,
            const unsigned block_idx, const unsigned thread_idx) {
          auto float_rgb =
              integrate_pixel(items, intersectable, division, settings, info);

          reduce_assign_output(items.base, division, thread_idx, block_idx,
                               info.x, info.y, float_rgb);
        });
#endif

    if (val.show_progress) {
      ++progress_bar;
      progress_bar.display();
    }
  }

  if (val.show_progress) {
    progress_bar.done();
  }
}
} // namespace detail
} // namespace render
