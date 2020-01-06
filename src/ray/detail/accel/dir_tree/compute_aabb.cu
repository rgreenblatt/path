#include "ray/detail/accel/dir_tree/impl/compute_aabbs_impl.h"
#include "ray/detail/block_data.h"

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
__global__ void compute_aabbs_global(Span<const Eigen::Projective3f> transforms,
                                     unsigned num_transforms,
                                     Span<IdxAABB> aabbs,
                                     Span<const BoundingPoints> bounds,
                                     unsigned num_bounds) {
  compute_aabbs_impl(transforms, threadIdx.y + blockDim.y * blockIdx.y,
                     num_transforms, aabbs, bounds,
                     threadIdx.x + blockDim.x * blockIdx.x, num_bounds);
}

template <>
void compute_aabbs<ExecutionModel::GPU>(
    Span<const Eigen::Projective3f> transforms, unsigned num_transforms,
    Span<IdxAABB> aabbs, Span<const BoundingPoints> bounds,
    unsigned num_bounds) {
  unsigned bounds_block_size = 64;
  unsigned transform_block_size = 4;

  dim3 grid(num_blocks(num_bounds, bounds_block_size),
            num_blocks(num_transforms, transform_block_size));
  dim3 block(bounds_block_size, transform_block_size);

  compute_aabbs_global<<<grid, block>>>(transforms, num_transforms, aabbs,
                                        bounds, num_bounds);
}
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
