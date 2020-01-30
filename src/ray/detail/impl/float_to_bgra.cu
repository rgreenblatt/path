#include "lib/bgra.h"
#include "lib/span_convertable_device_vector.h"
#include "lib/span_convertable_vector.h"
#include "lib/timer.h"
#include "ray/detail/block_data.h"
#include "ray/detail/impl/float_to_bgra.h"
#include "ray/detail/render_impl.h"
#include "ray/detail/render_impl_utils.h"

namespace ray {
namespace detail {
__global__ void float_to_bgra_global(unsigned x_dim, unsigned y_dim,
                                     unsigned super_sampling_rate,
                                     Span<const scene::Color> colors,
                                     Span<BGRA> bgra) {
  unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned y = threadIdx.y + blockIdx.y * blockDim.y;

  float_to_bgra_impl(x, y, x_dim, y_dim, super_sampling_rate, colors, bgra);
}

template <ExecutionModel execution_model>
void RendererImpl<execution_model>::float_to_bgra(
    BGRA *pixels, Span<const scene::Color> colors) {
  const unsigned x_block_size = 32;
  const unsigned y_block_size = 8;
  dim3 grid(num_blocks(real_x_dim_, x_block_size),
            num_blocks(real_y_dim_, y_block_size));
  dim3 block(x_block_size, y_block_size);

  Timer convert_to_bgra_timer;

  float_to_bgra_global<<<grid, block>>>(real_x_dim_, real_y_dim_,
                                        super_sampling_rate_, colors, bgra_);

  if (show_times_) {
    convert_to_bgra_timer.report("convert to bgra");
  }

  CUDA_ERROR_CHK(cudaDeviceSynchronize());

  Timer copy_bgra_timer;

  thrust::copy(bgra_.begin(), bgra_.end(), pixels);

  if (show_times_) {
    copy_bgra_timer.report("copy bgra");
  }
}

template class RendererImpl<ExecutionModel::GPU>;
} // namespace detail
} // namespace ray
