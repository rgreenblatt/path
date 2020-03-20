#ifndef CPU_ONLY_BUILD
#include "lib/cuda/reduce.h"
#include "render/detail/impl/intensities_impl.h"
#include "render/detail/impl/render_impl.h"

#include <cli/ProgressBar.hpp>

namespace render {
namespace detail {
template <intersect::accel::AccelRef MeshAccel,
          intersect::accel::AccelRef TriAccel, LightSamplerRef L,
          DirSamplerRef D, TermProbRef T, rng::RngRef R>
__global__ void
intensities_global(const ComputationSettings &settings, unsigned start_blocks,
                   const WorkDivision division, unsigned x_dim, unsigned y_dim,
                   unsigned samples_per, const MeshAccel mesh_accel,
                   Span<const TriAccel> tri_accels, const L light_sampler,
                   const D direction_sampler, const T term_prob, const R rng,
                   Span<BGRA> bgras, Span<Eigen::Array3f>,
                   Span<const scene::TriangleData> triangle_data,
                   Span<const material::Material> materials,
                   const Eigen::Affine3f film_to_world) {
  assert(division.num_sample_blocks == 1);

  const unsigned block_idx = blockIdx.x + start_blocks;
  const unsigned thread_idx = threadIdx.x;
  const unsigned block_dim = blockDim.x;

  const unsigned block_idx_sample = block_idx % division.num_sample_blocks;
  const unsigned block_idx_pixel = block_idx / division.num_sample_blocks;
  const unsigned block_idx_x = block_idx_pixel % division.num_x_blocks;
  const unsigned block_idx_y = block_idx_pixel / division.num_x_blocks;

  unsigned work_idx = division.samples_per_thread * thread_idx;

  unsigned sample_block_size = samples_per / division.num_sample_blocks;

  const unsigned work_idx_sample = work_idx % sample_block_size;
  const unsigned work_idx_pixel = work_idx / sample_block_size;
  const unsigned work_idx_x = work_idx_pixel % division.x_block_size;
  const unsigned work_idx_y = work_idx_pixel / division.x_block_size;

  const unsigned start_sample =
      work_idx_sample + block_idx_sample * sample_block_size;
  const unsigned end_sample = start_sample + division.samples_per_thread;
  const unsigned x = work_idx_x + block_idx_x * division.x_block_size;
  const unsigned y = work_idx_y + block_idx_y * division.y_block_size;

  if (x >= x_dim || y >= y_dim) {
    return;
  }

  auto intensity = intensities_impl(
      x, y, start_sample, end_sample, settings, x_dim, y_dim, samples_per,
      mesh_accel, tri_accels, light_sampler, direction_sampler, term_prob, rng,
      triangle_data, materials, film_to_world);

  // below reduction assumes this is the case
  assert(division.num_sample_blocks == 1);

  auto compute_bgras = [&](const auto &reduce_func, unsigned idx) {
    Eigen::Array3f totals;
    for (unsigned axis = 0; axis < 3; axis++) {
      totals[axis] = reduce_func(intensity[axis]);
    }
    if (idx == 0) {
      bgras[x + y * x_dim] = intensity_to_bgr(totals / samples_per);
    }
  };

  auto add = [](auto lhs, auto rhs) { return lhs + rhs; };

  switch (division.sample_reduction_strategy) {
  case ReductionStrategy::Block:
    compute_bgras(
        [&](const float v) {
          return block_reduce(v, add, 0.0f, thread_idx, block_dim);
        },
        thread_idx);
    break;
  case ReductionStrategy::Warp:
    compute_bgras([&](const float v) { return warp_reduce(v, add); },
                  thread_idx % warpSize);
    break;
  case ReductionStrategy::Thread:
    compute_bgras([&](const float v) { return v; }, 0);
    break;
  }
}

template <intersect::accel::AccelRef MeshAccel,
          intersect::accel::AccelRef TriAccel, LightSamplerRef L,
          DirSamplerRef D, TermProbRef T, rng::RngRef R>
void intensities(const ComputationSettings &settings,
                 const WorkDivision &division, unsigned samples_per,
                 unsigned x_dim, unsigned y_dim, const MeshAccel &mesh_accel,
                 Span<const TriAccel> tri_accels, const L &light_sampler,
                 const D &direction_sampler, const T &term_prob, const R &rng,
                 Span<BGRA> pixels, Span<Eigen::Array3f> intensities,
                 Span<const scene::TriangleData> triangle_data,
                 Span<const material::Material> materials,
                 const Eigen::Affine3f &film_to_world) {
  unsigned block_size = division.block_size;

  size_t total_size = size_t(samples_per) * x_dim * y_dim;
  size_t total_items_per_block = division.samples_per_thread * block_size;
  assert(total_size >= total_items_per_block);
  assert(total_size % total_items_per_block == 0);
  size_t total_grid =
      (total_size + total_items_per_block - 1) / total_items_per_block;

  size_t max_launch_size = 2 << 24;

  size_t num_launches = (total_size + max_launch_size - 1) / max_launch_size;
  size_t blocks_per = total_grid / num_launches;

  ProgressBar progress_bar(num_launches, 70);
  progress_bar.display();

  for (unsigned i = 0; i < num_launches; i++) {
    unsigned start = i * blocks_per;
    unsigned end = std::min((i + 1) * blocks_per, total_grid);
    unsigned grid = end - start;
    intensities_global<<<grid, block_size>>>(
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
#endif
