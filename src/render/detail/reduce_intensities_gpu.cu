#ifndef CPU_ONLY
#include "lib/assert.h"
#include "render/detail/reduce_assign_output.h"
#include "render/detail/reduce_intensities_gpu.h"
#include "work_division/work_division.h"
#include "work_division/work_division_impl.h"

#include "data_structure/copyable_to_vec.h"

namespace render {
namespace detail {
__global__ void reduce_intensities_global(
    bool output_as_bgra, unsigned reduction_factor, unsigned samples_per,
    unsigned x_dim, const work_division::WorkDivision division,
    Span<const Eigen::Array3f> intensities_in,
    Span<Eigen::Array3f> intensities_out, Span<BGRA> bgras) {
  const unsigned block_idx = blockIdx.x;
  const unsigned thread_idx = threadIdx.x;

  debug_assert(blockDim.x == division.block_size());

  auto [start_sample, end_sample, x, y, exit] =
      division.get_thread_info(block_idx, thread_idx, x_dim, 1);

  if (exit) {
    return;
  }

  Eigen::Array3f total = Eigen::Array3f::Zero();
  for (unsigned i = start_sample; i < end_sample; ++i) {
    total += intensities_in[i + x * reduction_factor];
  }

  reduce_assign_output(thread_idx, block_idx, output_as_bgra, x, 0, 0, total,
                       bgras, intensities_out, division, samples_per);
}

DeviceVector<Eigen::Array3f> *reduce_intensities_gpu(
    bool output_as_bgra, unsigned reduction_factor, unsigned samples_per,
    DeviceVector<Eigen::Array3f> *intensities_in,
    DeviceVector<Eigen::Array3f> *intensities_out, Span<BGRA> bgras) {
  while (reduction_factor != 1) {
    always_assert(intensities_in->size() % reduction_factor == 0);
    unsigned x_dim = intensities_in->size() / reduction_factor;
    work_division::WorkDivision division(
        {.block_size = 256,
         .target_x_block_size = 256,
         .force_target_samples = false,
         .forced_target_samples_per_thread = 1,
         .base_num_threads = 16384,
         .samples_per_thread_scaling_power = 0.5f,
         .max_samples_per_thread = 8},
        reduction_factor, x_dim, 1);
    intensities_out->resize(x_dim * division.num_sample_blocks());

    reduce_intensities_global<<<division.total_num_blocks(),
                                division.block_size()>>>(
        output_as_bgra && division.num_sample_blocks() == 1, reduction_factor,
        samples_per, x_dim, division, *intensities_in, *intensities_out, bgras);

    CUDA_ERROR_CHK(cudaDeviceSynchronize());
    CUDA_ERROR_CHK(cudaGetLastError());

    reduction_factor = division.num_sample_blocks();

    std::swap(intensities_in, intensities_out);
  }

  return intensities_in;
}
} // namespace detail
} // namespace render
#endif
