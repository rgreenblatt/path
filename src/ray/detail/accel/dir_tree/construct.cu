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
  better_than_no_split_.first->resize(num_groups());
  better_than_no_split_.second->resize(num_groups());
  thrust::fill(better_than_no_split_.second->begin(),
               better_than_no_split_.second->end(), true);
  while (num_groups() != 0) {
    better_than_no_split_.first->resize(num_groups());

    x_edges_keys_.resize(current_edges_->size());
    y_edges_keys_.resize(other_edges_->size());
    z_keys_.resize(sorted_by_z_min_.first->size());

    fill_keys();

    scan_edges();

    find_best_edges();

    test_splits();

    filter_others();

    setup_groups();

    std::swap(current_edges_, other_edges_new_);
    std::swap(other_edges_, other_edges_new_);
    std::swap(current_edges_keys_, other_edges_keys_);
    std::swap(sorted_by_z_min_.first, sorted_by_z_min_.second);
    std::swap(sorted_by_z_max_.first, sorted_by_z_max_.second);
    std::swap(axis_groups_.first, axis_groups_.second);
    std::swap(open_mins_before_group_.first, open_mins_before_group_.second);
    std::swap(num_per_group_.first, num_per_group_.second);
    std::swap(better_than_no_split_.first, better_than_no_split_.second);

    is_x_ = !is_x_;
  }
}

template class DirTreeGeneratorImpl<ExecutionModel::CPU>;
template class DirTreeGeneratorImpl<ExecutionModel::GPU>;
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
