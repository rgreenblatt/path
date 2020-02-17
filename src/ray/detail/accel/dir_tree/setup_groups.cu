#include "lib/cuda/utils.h"
#include "lib/span_convertable_device_vector.h"
#include "lib/span_convertable_vector.h"
#include "ray/detail/accel/dir_tree/dir_tree_generator_impl.h"
#include "ray/detail/accel/dir_tree/group.h"
#include "ray/detail/accel/dir_tree/left_right_counts.h"

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
template <ExecutionModel execution_model>
void DirTreeGeneratorImpl<execution_model>::setup_groups() {
  unsigned size = better_than_no_split_.first->size();
  unsigned new_num_groups = num_groups_inclusive_[size - 1];
  axis_groups_.second->resize_all(new_num_groups);
  num_per_group_.second->resize(new_num_groups);
  better_than_no_split_.second->resize(new_num_groups);
  current_edges_min_max_new_->resize(new_num_groups);
  other_edges_min_max_new_->resize(new_num_groups);

  std::array<Span<const unsigned>, 2> new_indexes_arr = {new_edge_indexes_,
                                                         new_z_min_indexes_};
  Span<const unsigned> outputs_inclusive = current_edge_outputs_inclusive_;
  Span<const unsigned> edges_groups = current_edges_groups();
  Span<unsigned> edges_new_groups = current_edges_new_groups();
  Span<const uint8_t> is_mins = current_edges_->is_mins();

  Span<const float> edge_values = current_edges_->values();

  Span<const unsigned> old_num_per_group = num_per_group_.first.get();
  Span<const uint8_t> old_better_than_no_split =
      better_than_no_split_.first.get();
  Span<const unsigned> num_groups_inclusive = num_groups_inclusive_;
  Span<const unsigned> z_outputs_inclusive = z_outputs_inclusive_;
  Span<const std::array<float, 2>> old_current_edges_min_max =
      current_edges_min_max_.get();
  Span<const std::array<float, 2>> old_other_edges_min_max =
      other_edges_min_max_.get();

  Span<unsigned> new_num_per_group = num_per_group_.second.get();
  Span<uint8_t> new_better_than_no_split = better_than_no_split_.second.get();
  Span<const unsigned> best_edges_idxs = best_edges_.idxs();
  Span<const unsigned> starts_inclusive = starts_inclusive_;
  std::array<Span<const unsigned>, 2> old_groups_arr = {
      other_edges_groups(), axis_groups_.first.get()[2]};
  std::array<Span<unsigned>, 2> new_groups_arr = {other_edges_new_groups(),
                                                  axis_groups_.second.get()[2]};
  Span<std::array<float, 2>> new_current_edges_min_max =
      current_edges_min_max_new_.get();
  Span<std::array<float, 2>> new_other_edges_min_max =
      other_edges_min_max_new_.get();

  unsigned node_offset = node_offset_;
  unsigned new_node_offset = node_offset + num_groups();
  nodes_.resize(new_node_offset);
  Span<DirTreeNode> nodes = nodes_;
  unsigned output_values_offset = output_values_offset_;

  auto start_counting_it = thrust::make_counting_iterator(0u);
  thrust::for_each(
      thrust_data_[0].execution_policy(), start_counting_it,
      start_counting_it + size, [=] __host__ __device__(unsigned group_idx) {
        unsigned num_groups_prev =
            get_previous(group_idx, num_groups_inclusive);

        unsigned best_edge_idx = best_edges_idxs[group_idx];

        if (num_groups_inclusive[group_idx] != num_groups_prev) {
          bool is_best_edge_min = is_mins[best_edge_idx];
          unsigned start_inclusive = starts_inclusive[best_edge_idx];
          auto [start, end] = group_start_end(group_idx, edges_groups);
          assert(end > start);
          unsigned index_in_group = best_edge_idx - start;

          auto [num_left, num_right] = left_right_counts(
              index_in_group, start_inclusive, starts_inclusive[end - 1],
              is_best_edge_min, old_num_per_group[group_idx]);

          bool using_left = num_left > 0;
          bool using_right = num_right > 0;
          unsigned left_output_idx = num_groups_prev;
          unsigned right_output_idx =
              using_left ? num_groups_prev + 1 : num_groups_prev;

          for (uint8_t i = 0; i < 2; i++) {
            auto old_groups = old_groups_arr[i];
            auto new_indexes = new_indexes_arr[i];
            auto new_groups = new_groups_arr[i];
            auto [start, end] = group_start_end(group_idx, old_groups);
            unsigned size = group_size(group_idx, old_groups);
            assert(size != 0); // TODO
            unsigned start_new_indexes = start * 2;
            if (using_left) {
              new_groups[left_output_idx] =
                  new_indexes[start_new_indexes + size - 1];
            }
            if (using_right) {
              new_groups[right_output_idx] =
                  new_indexes[start_new_indexes + 2 * size - 1];
            }
            if (using_left && using_right) {
              assert(new_groups[right_output_idx] !=
                     new_groups[left_output_idx]);
            }
          }

          // predicting left and right idxs
          // 0 is a end node of size 0
          unsigned left_idx =
              using_left ? new_node_offset + left_output_idx : 0;
          unsigned right_idx =
              using_right ? new_node_offset + right_output_idx : 0;
          float edge_value = edge_values[best_edge_idx];
          nodes[node_offset + group_idx] =
              DirTreeNode(edge_value, left_idx, right_idx);

          if (using_left) {
            edges_new_groups[left_output_idx] =
                ((best_edge_idx + (is_best_edge_min ? 0 : 1)) - start) +
                get_previous(group_idx, outputs_inclusive);
            new_num_per_group[left_output_idx] = num_left;
            new_better_than_no_split[left_output_idx] =
                old_better_than_no_split[group_idx];
            // left side to edge
            new_current_edges_min_max[left_output_idx] = {
                old_current_edges_min_max[group_idx][0], edge_value};
            new_other_edges_min_max[left_output_idx] =
                old_other_edges_min_max[group_idx];
          }

          if (using_right) {
            edges_new_groups[right_output_idx] = outputs_inclusive[group_idx];
            new_num_per_group[right_output_idx] = num_right;
            new_better_than_no_split[right_output_idx] =
                old_better_than_no_split[group_idx];
            // edge to right side
            new_current_edges_min_max[right_output_idx] = {
                edge_value, old_current_edges_min_max[group_idx][1]};
            new_other_edges_min_max[right_output_idx] =
                old_other_edges_min_max[group_idx];
          }
          if (using_left && using_right &&
              edges_new_groups[right_output_idx] ==
                  edges_new_groups[left_output_idx]) {
            assert(outputs_inclusive[group_idx] !=
                   get_previous(group_idx, outputs_inclusive));
            // hacky approach to ensure 1 remains in group
            edges_new_groups[left_output_idx]--;
          }
        } else {
          // z group sizes
          unsigned start = get_previous(group_idx, z_outputs_inclusive) +
                           output_values_offset; // offset...
          unsigned size = group_size(group_idx, old_groups_arr[1]);
          nodes[node_offset + group_idx] = DirTreeNode(start, start + size);
        }
      });

  node_offset_ = new_node_offset;
  output_values_offset_ = min_sorted_values_.size();
}

template class DirTreeGeneratorImpl<ExecutionModel::CPU>;
template class DirTreeGeneratorImpl<ExecutionModel::GPU>;
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
