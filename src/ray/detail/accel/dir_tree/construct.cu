#include "lib/cuda/utils.h"
#include "lib/span_convertable_device_vector.h"
#include "lib/span_convertable_vector.h"
#include "ray/detail/accel/dir_tree/dir_tree_generator_impl.h"

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
template <ExecutionModel execution_model>
void DirTreeGeneratorImpl<execution_model>::construct() {
  while (num_groups() != 0) {
    better_than_no_split_.first->resize(num_groups());
    better_than_no_split_.second->resize(num_groups(), 1);

    x_edges_keys_.resize(current_edges_->size());
    y_edges_keys_.resize(other_edges_->size());
    z_keys_.resize(sorted_by_z_min_.first->size());

    fill_keys();

    scan_edges();

    find_best_edges();

    test_splits();

    filter_others();

    setup_groups();

    is_x_ = !is_x_;
  }
}

template class DirTreeGeneratorImpl<ExecutionModel::CPU>;
template class DirTreeGeneratorImpl<ExecutionModel::GPU>;
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
