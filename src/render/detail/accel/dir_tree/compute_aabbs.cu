#include "lib/span_convertable_device_vector.h"
#include "lib/span_convertable_vector.h"
#include "ray/detail/accel/dir_tree/dir_tree_generator_impl.h"
#include "ray/detail/accel/dir_tree/impl/compute_aabbs_impl.h"
#include "ray/detail/block_data.h"

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
__global__ void
compute_aabbs_global(SpanSized<const Eigen::Projective3f> transforms,
                     Span<IdxAABB> aabbs,
                     SpanSized<const BoundingPoints> bounds) {
  compute_aabbs_impl(transforms, threadIdx.y + blockDim.y * blockIdx.y, aabbs,
                     bounds, threadIdx.x + blockDim.x * blockIdx.x);
}

template <ExecutionModel execution_model>
void DirTreeGeneratorImpl<execution_model>::compute_aabbs() {
  unsigned bounds_block_size = 64;
  unsigned transform_block_size = 4;

  dim3 grid(num_blocks(bounds_.size(), bounds_block_size),
            num_blocks(transforms_.size(), transform_block_size));
  dim3 block(bounds_block_size, transform_block_size);

  compute_aabbs_global<<<grid, block>>>(transforms_, aabbs_, bounds_);

  // TODO: Needed???
  CUDA_ERROR_CHK(cudaDeviceSynchronize());
}

template class DirTreeGeneratorImpl<ExecutionModel::GPU>;
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
