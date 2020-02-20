#include "render/detail/impl/compute_intensities_impl.h"
#include "render/detail/impl/dispatch_compute_intensities.h"
#include "render/detail/renderer_impl.h"

namespace render {
namespace detail {
template <ExecutionModel execution_model, typename Accel, typename LightSampler,
          typename DirSampler, typename TermProb>
void compute_intensities(const WorkDivision &division, unsigned samples_per,
                         unsigned x_dim, unsigned y_dim, unsigned block_size,
                         const Accel &accel, const LightSampler &light_sampler,
                         const DirSampler &direction_sampler,
                         const TermProb &term_prob,
                         Span<Eigen::Array3f> intensities,
                         Span<const scene::TriangleData> triangle_data,
                         Span<const scene::Material> materials,
                         const Eigen::Affine3f &film_to_world) {
  unsigned grid =
      ceil_divide(samples_per * x_dim * y_dim, division.sample_block_size *
                                                   division.x_block_size *
                                                   division.y_block_size);

#pragma omp parallel for collapse(2) schedule(dynamic, 16)
  for (unsigned block_idx = 0; block_idx < grid; block_idx++) {
    for (unsigned thread_idx = 0; thread_idx < block_size; thread_idx++) {
      compute_intensities_impl(block_idx, thread_idx, block_size, division,
                               x_dim, y_dim, accel, light_sampler,
                               direction_sampler, term_prob, intensities,
                               triangle_data, materials, film_to_world);
    }
  }
}

template class RendererImpl<ExecutionModel::CPU>;
} // namespace detail
} // namespace render
