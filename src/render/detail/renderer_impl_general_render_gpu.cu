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
template <intersect::Intersectable I,
          intersectable_scene::SceneRef<typename I::InfoType> S,
          LightSamplerRef<typename S::B> L, DirSamplerRef<typename S::B> D,
          TermProbRef T, rng::RngRef R>
__global__ void integrate_image_global(
    bool output_as_bgra, const integrate::RenderingEquationSettings settings,
    unsigned start_blocks, const WorkDivision division, unsigned x_dim,
    unsigned y_dim, unsigned samples_per, const I &intersectable,
    const S &scene, const L light_sampler, const D direction_sampler,
    const T term_prob, const R rng, Span<BGRA> bgras,
    Span<Eigen::Array3f> intensities, const Eigen::Affine3f film_to_world) {
  const unsigned block_idx = blockIdx.x + start_blocks;
  const unsigned thread_idx = threadIdx.x;

  debug_assert(blockDim.x == division.block_size());

  auto [start_sample, end_sample, x, y, exit] =
      division.get_thread_info(block_idx, thread_idx, x_dim, y_dim);

  if (exit) {
    return;
  }

  auto intensity = integrate_pixel(
      x, y, start_sample, end_sample, settings, x_dim, y_dim, intersectable,
      scene, light_sampler, direction_sampler, term_prob, rng, film_to_world);

  reduce_assign_output(thread_idx, block_idx, output_as_bgra, x, y, y_dim,
                       intensity, bgras, intensities, division, samples_per);
}

template <>
template <intersectable_scene::IntersectableScene S,
          LightSamplerRef<typename S::B> L, DirSamplerRef<typename S::B> D,
          TermProbRef T, rng::RngRef R>
void IntegrateImage<ExecutionModel::GPU>::run(
    bool output_as_bgra, const GeneralSettings &settings, bool show_progress,
    const WorkDivision &division, unsigned samples_per, unsigned x_dim,
    unsigned y_dim, S &scene, const L &light_sampler,
    const D &direction_sampler, const T &term_prob, const R &rng,
    Span<BGRA> pixels, Span<Eigen::Array3f> intensities,
    const Eigen::Affine3f &film_to_world) {
  size_t total_grid = division.total_num_blocks();

  size_t max_launch_size = settings.computation_settings.max_blocks_per_launch;

  size_t num_launches = ceil_divide(total_grid, max_launch_size);
  size_t blocks_per = total_grid / num_launches;

  ProgressBar progress_bar(num_launches, 70);
  if (show_progress) {
    progress_bar.display();
  }

  for (unsigned i = 0; i < num_launches; i++) {
    unsigned start = i * blocks_per;
    unsigned end = std::min((i + 1) * blocks_per, total_grid);
    unsigned grid = end - start;

    // TODO
    integrate_image_global<<<grid, division.block_size()>>>(
        output_as_bgra, settings.rendering_equation_settings, start, division,
        x_dim, y_dim, samples_per, scene.intersectable(), scene.scene(),
        light_sampler, direction_sampler, term_prob, rng, pixels, intensities,
        film_to_world);

    CUDA_ERROR_CHK(cudaDeviceSynchronize());
    CUDA_ERROR_CHK(cudaGetLastError());

    if (show_progress) {
      ++progress_bar;
      progress_bar.display();
    }
  }

  if (show_progress) {
    progress_bar.done();
  }
}
} // namespace detail

template class Renderer::Impl<ExecutionModel::GPU>;
} // namespace render
#endif
