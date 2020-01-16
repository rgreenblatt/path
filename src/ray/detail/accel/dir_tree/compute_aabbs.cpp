#include "lib/span_convertable_vector.h"
#include "ray/detail/accel/dir_tree/dir_tree_generator.h"
#include "ray/detail/accel/dir_tree/impl/compute_aabbs_impl.h"

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
template <ExecutionModel execution_model>
void DirTreeGenerator<execution_model>::compute_aabbs() {
#pragma omp parallel for collapse(2) schedule(dynamic, 16)
  for (unsigned transform_idx = 0; transform_idx < transforms_.size();
       transform_idx++) {
    for (unsigned bound_idx = 0; bound_idx < bounds_.size(); bound_idx++) {
      compute_aabbs_impl(transforms_, transform_idx, aabbs_, bounds_,
                         bound_idx);
    }
  }
}
template class DirTreeGenerator<ExecutionModel::CPU>;
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
