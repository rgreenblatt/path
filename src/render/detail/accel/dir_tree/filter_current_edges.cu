#include "ray/detail/accel/dir_tree/dir_tree_generator_impl.h"
#include "ray/detail/accel/dir_tree/group.h"

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
template <ExecutionModel execution_model>
void DirTreeGeneratorImpl<execution_model>::filter_current_edges() {
  Span<const unsigned> keys = current_edges_keys_.get();
  Span<const float> values = current_edges_->values();
  Span<const float> other_mins = current_edges_->other_mins();
  Span<const float> other_maxs = current_edges_->other_maxs();
  Span<const uint8_t> is_mins = current_edges_->is_mins();
  Span<const unsigned> outputs_inclusive = current_edge_outputs_inclusive_;
  Span<const unsigned> groups = current_edges_groups();

  current_edges_new_->resize_all(
      current_edge_outputs_inclusive_[current_edge_outputs_inclusive_.size() -
                                      1]);

  Span<float> new_values = current_edges_new_->values();
  Span<float> new_other_mins = current_edges_new_->other_mins();
  Span<float> new_other_maxs = current_edges_new_->other_maxs();
  Span<uint8_t> new_is_mins = current_edges_new_->is_mins();

  auto start_it = thrust::make_counting_iterator(0u);
  thrust::for_each(thrust_data_[0].execution_policy(), start_it,
                   start_it + current_edges_->size(),
                   [=] __host__ __device__(unsigned i) {
                     unsigned key = keys[i];
                     unsigned previous = get_previous(key, outputs_inclusive);
                     if (previous != outputs_inclusive[key]) {
                       auto [start, end] = group_start_end(key, groups);
                       unsigned index_in_group = i - start;
                       unsigned new_idx = index_in_group + previous;
                       new_values[new_idx] = values[i];
                       new_other_mins[new_idx] = other_mins[i];
                       new_other_maxs[new_idx] = other_maxs[i];
                       new_is_mins[new_idx] = is_mins[i];
                     }
                   });
}

template class DirTreeGeneratorImpl<ExecutionModel::GPU>;
template class DirTreeGeneratorImpl<ExecutionModel::CPU>;
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
