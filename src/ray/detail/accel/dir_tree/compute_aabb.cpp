#include "ray/detail/accel/dir_tree/impl/compute_aabbs_impl.h"
#include "ray/detail/block_data.h"

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
template <>
void compute_aabbs<ExecutionModel::CPU>(
    Span<const Eigen::Projective3f> transforms, unsigned num_transforms,
    Span<IdxAABB> aabbs, Span<const BoundingPoints> bounds,
    unsigned num_bounds) {
#pragma omp parallel for collapse(2) schedule(dynamic, 16)
  for (unsigned transform_idx = 0; transform_idx < num_transforms;
       transform_idx++) {
    for (unsigned bound_idx = 0; bound_idx < num_bounds; bound_idx++) {
      compute_aabbs_impl(transforms, transform_idx, num_transforms, aabbs,
                         bounds, bound_idx, num_bounds);
    }
  }
}
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
