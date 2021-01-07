#ifndef CPU_ONLY
#include "lib/assert.h"
#include "lib/integer_division_utils.h"
#include "render/detail/general_render_impl.h"
#include "render/detail/integrate_image.h"
#include "render/detail/integrate_pixel.h"
#include "render/detail/reduce_assign_output.h"
#include "render/detail/renderer_impl.h"
#include "work_division/work_division_impl.h"

#include <cli/ProgressBar.hpp>

namespace render {
namespace detail {
// could remove:
//  - block size (from division)
//  - samples_per
//  - rng sequence_gen samples_per
template <ExactSpecializationOf<IntegrateImageItems> Items,
          intersect::IntersectableForInfoType<typename Items::InfoType> I>
__global__ void
integrate_image_global(unsigned start_blocks,
                       const integrate::RenderingEquationSettings settings,
                       const I intersectable, const Items items) {
  const unsigned block_idx = blockIdx.x + start_blocks;
  const unsigned thread_idx = threadIdx.x;

  debug_assert(blockDim.x == items.base.division.block_size());

  auto [info, exit] = items.base.division.get_thread_info(
      block_idx, thread_idx, items.base.x_dim, items.base.y_dim);

  if (exit) {
    return;
  }

  auto intensity = integrate_pixel(info, settings, intersectable, items);

  reduce_assign_output(thread_idx, block_idx, items.base, info.x, info.y,
                       intensity);
}

template <> template <ExactSpecializationOf<IntegrateImageInputs> Inp>
requires Inp::I::individually_intersectable void
IntegrateImage<ExecutionModel::GPU>::run_individual(Inp inp) {
  unsigned total_grid = inp.items.base.division.total_num_blocks();

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
    unsigned grid = end - start;

    integrate_image_global<<<grid, inp.items.base.division.block_size()>>>(
        start, inp.settings.rendering_equation_settings, inp.intersector,
        inp.items);

    CUDA_ERROR_CHK(cudaDeviceSynchronize());
    CUDA_ERROR_CHK(cudaGetLastError());

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

template class Renderer::Impl<ExecutionModel::GPU>;
} // namespace render
#endif
