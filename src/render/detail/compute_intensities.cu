#include "render/detail/impl/compute_intensities_impl.h"
#include "render/detail/impl/dispatch_compute_intensities.h"
#include "render/detail/renderer_impl.h"

namespace render {
namespace detail {
template <typename Accel, typename LightSampler, typename DirSampler,
          typename TermProb>
__global__ void compute_intensities_global(
    const WorkDivision division, unsigned x_dim, unsigned y_dim,
    const Accel accel, const LightSampler light_sampler,
    const DirSampler direction_sampler, const TermProb term_prob,
    Span<Eigen::Array3f> intensities,
    Span<const scene::TriangleData> triangle_data,
    Span<const scene::Material> materials,
    const Eigen::Affine3f film_to_world) {
  compute_intensities_impl(blockIdx.x, threadIdx.x, blockDim.x, division, x_dim,
                           y_dim, accel, light_sampler, direction_sampler,
                           term_prob, intensities, triangle_data, materials,
                           film_to_world);
}

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

  compute_intensities_global<<<grid, block_size>>>(
      division, x_dim, y_dim, accel, light_sampler, direction_sampler,
      term_prob, intensities, triangle_data, materials, film_to_world);

  CUDA_ERROR_CHK(cudaDeviceSynchronize());
}

template class RendererImpl<ExecutionModel::GPU>;
} // namespace detail
} // namespace render
