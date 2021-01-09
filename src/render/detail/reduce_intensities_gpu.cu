#ifndef CPU_ONLY
#include "lib/assert.h"
#include "render/detail/integrate_image_base_items.h"
#include "render/detail/reduce_assign_output.cuh"
#include "render/detail/reduce_intensities_gpu.h"
#include "work_division/kernel_launch.h"
#include "work_division/kernel_launch_impl_gpu.cuh"
#include "work_division/work_division.h"

#include "data_structure/copyable_to_vec.h"

namespace render {
namespace detail {
DeviceVector<Eigen::Array3f> *reduce_intensities_gpu(
    bool output_as_bgra, unsigned reduction_factor, unsigned samples_per,
    DeviceVector<Eigen::Array3f> *intensities_in,
    DeviceVector<Eigen::Array3f> *intensities_out, Span<BGRA> bgras) {
  while (reduction_factor != 1) {
    always_assert(intensities_in->size() % reduction_factor == 0);
    unsigned x_dim = intensities_in->size() / reduction_factor;
    // TODO: reduce division settings
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

    IntegrateImageBaseItems items{
        .output_as_bgra = output_as_bgra && division.num_sample_blocks() == 1,
        .samples_per = samples_per,
        .pixels = bgras,
        .intensities = *intensities_out};

    Span<const Eigen::Array3f> in_span = *intensities_in;

    work_division::KernelLaunch<ExecutionModel::GPU>::run(
        division, 0, division.total_num_blocks(),
        [=] __device__(const work_division::WorkDivision &division,
                       const work_division::GridLocationInfo &info,
                       const unsigned block_idx, const unsigned thread_idx) {
          auto [start_sample, end_sample, x, y] = info;

          Eigen::Array3f total = Eigen::Array3f::Zero();
          for (unsigned i = start_sample; i < end_sample; ++i) {
            total += in_span[i + x * reduction_factor];
          }

          reduce_assign_output(items, division, thread_idx, block_idx, x, 0,
                               total);
        });

    reduction_factor = division.num_sample_blocks();

    std::swap(intensities_in, intensities_out);
  }

  return intensities_in;
}
} // namespace detail
} // namespace render
#endif
