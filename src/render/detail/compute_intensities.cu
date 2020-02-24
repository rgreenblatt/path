#include "render/detail/impl/compute_intensities_impl.h"
#include "render/detail/impl/render.h"

namespace render {
namespace detail {
template <intersect::accel::AccelRef A, LightSamplerRef L, DirSamplerRef D,
          TermProbRef T, rng::RngRef R>
__global__ void compute_intensities_global(
    const WorkDivision division, unsigned x_dim, unsigned y_dim,
    unsigned samples_per, const A accel, const L light_sampler,
    const D direction_sampler, const T term_prob, const R rng, Span<BGRA>,
    Span<Eigen::Array3f>, Span<const scene::TriangleData> triangle_data,
    Span<const material::Material> materials,
    const Eigen::Affine3f film_to_world) {
  const unsigned block_idx = blockIdx.x;
  const unsigned thread_idx = threadIdx.x;
  const unsigned block_dim = blockDim.x;

  const unsigned block_idx_sample = block_idx % division.num_sample_blocks;
  const unsigned block_idx_pixel = block_idx / division.num_sample_blocks;
  const unsigned block_idx_x = block_idx_pixel % division.num_x_blocks;
  const unsigned block_idx_y = block_idx_pixel / division.num_x_blocks;

  unsigned total_size_per_block = division.sample_block_size *
                                  division.x_block_size * division.y_block_size;
  assert(total_size_per_block % block_dim == 0);
  unsigned num_per_thread = total_size_per_block / block_dim;
  unsigned work_idx = num_per_thread * thread_idx;

  const unsigned work_idx_sample = work_idx % division.sample_block_size;
  const unsigned work_idx_pixel = work_idx / division.sample_block_size;
  const unsigned work_idx_x = work_idx_pixel % division.x_block_size;
  const unsigned work_idx_y = work_idx_pixel / division.x_block_size;

  const unsigned start_sample =
      work_idx_sample + block_idx_sample * division.sample_block_size;
  const unsigned end_sample = start_sample + num_per_thread;
  const unsigned x = work_idx_x + block_idx_x * division.x_block_size;
  const unsigned y = work_idx_y + block_idx_y * division.y_block_size;

  compute_intensities_impl(x, y, start_sample, end_sample, x_dim, y_dim,
                           samples_per, accel, light_sampler, direction_sampler,
                           term_prob, rng, triangle_data, materials,
                           film_to_world);

  // TODO: sum intensities
}

template <intersect::accel::AccelRef A, LightSamplerRef L, DirSamplerRef D,
          TermProbRef T, rng::RngRef R>
void compute_intensities(const WorkDivision &division, unsigned samples_per,
                         unsigned x_dim, unsigned y_dim, unsigned block_size,
                         const A &accel, const L &light_sampler,
                         const D &direction_sampler, const T &term_prob,
                         const R &rng, Span<BGRA> pixels,
                         Span<Eigen::Array3f> intensities,
                         Span<const scene::TriangleData> triangle_data,
                         Span<const material::Material> materials,
                         const Eigen::Affine3f &film_to_world) {
  unsigned grid =
      ceil_divide(samples_per * x_dim * y_dim, division.sample_block_size *
                                                   division.x_block_size *
                                                   division.y_block_size);

  compute_intensities_global<<<grid, block_size>>>(
      division, x_dim, y_dim, samples_per, accel, light_sampler,
      direction_sampler, term_prob, rng, pixels, intensities, triangle_data,
      materials, film_to_world);

  CUDA_ERROR_CHK(cudaDeviceSynchronize());
}

template class RendererImpl<ExecutionModel::GPU>;
} // namespace detail
} // namespace render
