#pragma once

#include "lib/assert.h"
#include "lib/integer_division_utils.h"
#include "render/detail/integrate_image.h"
#include "render/detail/integrate_pixel.h"
#include "render/detail/reduce_assign_output.cuh"
#include "work_division/kernel_launch_impl_gpu.cuh"

#include <cli/ProgressBar.hpp>

namespace render {
namespace detail {
template <> template <ExactSpecializationOf<IntegrateImageInputs> Inp>
requires Inp::I::individually_intersectable void
IntegrateImage<ExecutionModel::GPU>::run_individual(Inp inp) {
  unsigned total_grid = inp.division.total_num_blocks();

  unsigned max_launch_size =
      inp.settings.computation_settings.max_blocks_per_launch;

  unsigned num_launches = ceil_divide(total_grid, max_launch_size);
  unsigned blocks_per = total_grid / num_launches;

  ProgressBar progress_bar(num_launches, 70);
  if (inp.show_progress) {
    progress_bar.display();
  }

  for (unsigned i = 0; i < num_launches; i++) {
    unsigned start = i * blocks_per;
    unsigned end = std::min((i + 1) * blocks_per, total_grid);

    work_division::KernelLaunch<ExecutionModel::GPU>::run(
        inp.division, start, end,
        [=, items = inp.items, intersectable = inp.intersector,
         settings = inp.settings.rendering_equation_settings](
            const WorkDivision &division,
            const work_division::GridLocationInfo &info,
            const unsigned block_idx, const unsigned thread_idx) {
          auto intensity =
              integrate_pixel(items, intersectable, division, settings, info);

          reduce_assign_output(items.base, division, thread_idx, block_idx,
                               info.x, info.y, intensity);
        });

    if (inp.show_progress) {
      ++progress_bar;
      progress_bar.display();
    }
  }

  if (inp.show_progress) {
    progress_bar.done();
  }
}
} // namespace detail
} // namespace render