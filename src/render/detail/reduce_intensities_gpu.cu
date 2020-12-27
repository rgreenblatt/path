#ifndef CPU_ONLY
#include "lib/assert.h"
#include "render/detail/reduce_assign_output.h"
#include "render/detail/reduce_intensities_gpu.h"
#include "render/detail/work_division_impl.h"

namespace render {
namespace detail {
__global__ void reduce_intensities_global(
    bool output_as_bgra, unsigned reduction_factor, unsigned samples_per,
    unsigned x_dim, const WorkDivision division,
    Span<const Eigen::Array3f> intensities_in,
    Span<Eigen::Array3f> intensities_out, Span<BGRA> bgras) {
  const unsigned block_idx = blockIdx.x;
  const unsigned thread_idx = threadIdx.x;

  debug_assert(blockDim.x == division.block_size());

  auto [start_sample, end_sample, x, y] =
      division.get_thread_info(block_idx, thread_idx, reduction_factor);

  if (x >= x_dim) {
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
    const unsigned block_size = 256;
    const unsigned target_x_block_size = 256;
    const unsigned target_y_block_size = 1;
    const unsigned max_samples_per_thread = 16;
    const unsigned target_samples_per_thread = 8;
    WorkDivision division({block_size, target_x_block_size, target_y_block_size,
                           max_samples_per_thread, target_samples_per_thread},
                          reduction_factor, x_dim, 1);
    intensities_out->resize(x_dim * division.num_sample_blocks());

    reduce_intensities_global<<<division.total_num_blocks(),
                                division.block_size()>>>(
        output_as_bgra && division.num_sample_blocks() == 1, reduction_factor,
        samples_per, x_dim, division, *intensities_in, *intensities_out, bgras);
    reduction_factor = division.num_sample_blocks();

    std::swap(intensities_in, intensities_out);
  }

  return intensities_in;
}

} // namespace detail
} // namespace render
#endif
