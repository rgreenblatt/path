#include "lib/cuda/reduce.h"
#include "lib/info/debug_print.h"
#include "render/detail/impl/compute_intensities_impl.h"
#include "render/detail/impl/render.h"

#include <cli/ProgressBar.hpp>

namespace render {
namespace detail {
template <intersect::accel::AccelRef MeshAccel,
          intersect::accel::AccelRef TriAccel, LightSamplerRef L,
          DirSamplerRef D, TermProbRef T, rng::RngRef R>
__global__ void compute_intensities_global(
    const ComputationSettings &settings, unsigned start_blocks,
    const WorkDivision division, unsigned x_dim, unsigned y_dim,
    unsigned samples_per, const MeshAccel mesh_accel,
    Span<const TriAccel> tri_accels, const L light_sampler,
    const D direction_sampler, const T term_prob, const R rng, Span<BGRA> bgras,
    Span<Eigen::Array3f>, Span<const scene::TriangleData> triangle_data,
    Span<const material::Material> materials,
    const Eigen::Affine3f film_to_world) {
  const unsigned block_idx = blockIdx.x + start_blocks;
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

  auto intensity = compute_intensities_impl(
      x, y, start_sample, end_sample, settings, x_dim, y_dim, samples_per,
      mesh_accel, tri_accels, light_sampler, direction_sampler, term_prob, rng,
      triangle_data, materials, film_to_world);

  Eigen::Array3f totals;
  for (unsigned axis = 0; axis < 3; axis++) {
    totals[axis] = block_reduce(
        intensity[axis], [](float lhs, float rhs) { return lhs + rhs; }, 0.0f,
        threadIdx.x, blockDim.x);
  }
  if (threadIdx.x == 0) {
    bgras[x + y * x_dim] = intensity_to_bgr(totals / samples_per);
  }
}

template <intersect::accel::AccelRef MeshAccel,
          intersect::accel::AccelRef TriAccel, LightSamplerRef L,
          DirSamplerRef D, TermProbRef T, rng::RngRef R>
void compute_intensities(const ComputationSettings &settings,
                         const WorkDivision &division, unsigned samples_per,
                         unsigned x_dim, unsigned y_dim,
                         const MeshAccel &mesh_accel,
                         Span<const TriAccel> tri_accels,
                         const L &light_sampler, const D &direction_sampler,
                         const T &term_prob, const R &rng, Span<BGRA> pixels,
                         Span<Eigen::Array3f> intensities,
                         Span<const scene::TriangleData> triangle_data,
                         Span<const material::Material> materials,
                         const Eigen::Affine3f &film_to_world) {
  unsigned block_size = division.block_size;
  unsigned total_size = samples_per * x_dim * y_dim;
  unsigned total_grid =
      ceil_divide(samples_per * x_dim * y_dim, division.sample_block_size *
                                                   division.x_block_size *
                                                   division.y_block_size);

  unsigned max_launch_size = 2 << 24;

  unsigned num_launches = (total_size + max_launch_size - 1) / max_launch_size;
  unsigned blocks_per = total_grid / num_launches;

  ProgressBar progress_bar(num_launches, 70);
  progress_bar.display();

  for (unsigned i = 0; i < num_launches; i++) {
    unsigned start = i * blocks_per;
    unsigned end = std::min((i + 1) * blocks_per, total_grid);
    unsigned grid = end - start;
    compute_intensities_global<<<grid, block_size>>>(
        settings, start, division, x_dim, y_dim, samples_per, mesh_accel,
        tri_accels, light_sampler, direction_sampler, term_prob, rng, pixels,
        intensities, triangle_data, materials, film_to_world);

    CUDA_ERROR_CHK(cudaDeviceSynchronize());

    ++progress_bar;
    progress_bar.display();
  }

  progress_bar.done();
}

template class RendererImpl<ExecutionModel::GPU>;
} // namespace detail
} // namespace render
