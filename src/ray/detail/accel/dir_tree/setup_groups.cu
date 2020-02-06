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
  Span<const uint8_t> use_split_first = better_than_no_split_.first.get();
  Span<const uint8_t> use_split_second = better_than_no_split_.second.get();
  Span<const unsigned> z_groups = axis_groups_.second.get()[2];
  auto start_it = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0u),

      [=] __host__ __device__(unsigned i) -> uint64_t {
        bool using_split = !(use_split_first[i] || use_split_second[i]);
        std::array<uint32_t, 2> out = {
            using_split, using_split ? 0 : group_size(i, z_groups)};
        return *reinterpret_cast<uint64_t *>(&out);
      });

  unsigned size = better_than_no_split_.first->size();
  thrust::inclusive_scan(thrust_data_[0].execution_policy(), start_it,
                         start_it + size,
                         num_groups_num_z_done_before_.begin());

  uint64_t raw_last_value = num_groups_num_z_done_before_[size - 1];
  unsigned num_groups_split =
      (*reinterpret_cast<const std::array<uint32_t, 2> *>(&raw_last_value))[0];

  unsigned new_num_groups = num_groups_split * 2;
  axis_groups_.second->resize_all(new_num_groups);
  open_mins_before_group_.second->resize(new_num_groups);
  num_per_group_.second->resize(new_num_groups);
  better_than_no_split_.second->resize(new_num_groups);

  Span<const uint64_t> raw_vals = num_groups_num_z_done_before_;
  Span<const std::array<uint32_t, 2>> num_groups_num_z_done_before(
      reinterpret_cast<const std::array<uint32_t, 2> *>(raw_vals.data()),
      num_groups_num_z_done_before_.size());
  std::array<Span<const unsigned>, 2> new_indexes_arr = {new_edge_indexes_,
                                                         new_z_min_indexes_};
  Span<const unsigned> edges_groups = current_edges_groups();
  Span<unsigned> edges_new_groups = current_edges_new_groups();
  Span<const uint8_t> is_mins = current_edges_->is_mins();

#pragma message("AAAAAAHHHHH, this will change.....")
  Span<const float> edge_values = current_edges_->values();

  Span<const unsigned> old_open_mins_before_group =
      open_mins_before_group_.first.get();
  Span<const unsigned> old_num_per_group = num_per_group_.first.get();
  Span<const uint8_t> old_better_than_no_split =
      better_than_no_split_.first.get();

  Span<unsigned> new_open_mins_before_group =
      open_mins_before_group_.second.get();
  Span<unsigned> new_num_per_group = num_per_group_.second.get();
  Span<uint8_t> new_better_than_no_split = better_than_no_split_.second.get();
  Span<const BestEdge> best_edges = best_edges_;
  Span<const unsigned> starts_inclusive = starts_inclusive_;
  std::array<Span<const unsigned>, 2> old_groups_arr = {
      other_edges_groups(), axis_groups_.first.get()[2]};
  std::array<Span<unsigned>, 2> new_groups_arr = {other_edges_new_groups(),
                                                  axis_groups_.second.get()[2]};

  Span<DirTreeNode> nodes = nodes_;
  unsigned node_offset = node_offset_;
  unsigned new_node_offset = node_offset + num_groups();

  auto start_counting_it = thrust::make_counting_iterator(0u);
  thrust::for_each(
      thrust_data_[0].execution_policy(), start_counting_it,
      start_counting_it + size, [=] __host__ __device__(unsigned group_idx) {
        // TODO ...
        std::array<uint32_t, 2> previous_values =
            get_previous(group_idx, num_groups_num_z_done_before);

        unsigned best_edge_idx = best_edges[group_idx].idx;

        if (num_groups_num_z_done_before[group_idx][0] != previous_values[0]) {
          unsigned output_idx = previous_values[0] * 2;
          for (uint8_t i = 0; i < 2; i++) {
            auto old_groups = old_groups_arr[i];
            auto new_indexes = new_indexes_arr[i];
            auto new_groups = new_groups_arr[i];
            auto [start, end] = group_start_end(group_idx, old_groups);
            unsigned size = group_size(group_idx, old_groups);
            assert(size != 0); // TODO
            unsigned start_new_indexes = start * 2;
            new_groups[output_idx] = new_indexes[start_new_indexes + size - 1];
            new_groups[output_idx + 1] =
                new_indexes[start_new_indexes + 2 * size - 1];
          }

          // predicting left and right idxs
          unsigned left_idx = new_node_offset + output_idx;
          unsigned right_idx = left_idx + 1;
          nodes[node_offset + group_idx] =
              DirTreeNode(edge_values[best_edge_idx], left_idx, right_idx);

          bool is_best_edge_min = is_mins[best_edge_idx];
          unsigned start_inclusive = starts_inclusive[best_edge_idx];
          auto [start, end] = group_start_end(group_idx, edges_groups);
          edges_new_groups[output_idx] =
              best_edge_idx + (is_best_edge_min ? 0 : 1);
          edges_new_groups[output_idx + 1] = end;
          unsigned index_in_group = best_edge_idx - start;
          auto [num_left, num_right, open_mins_before_right] =
              left_right_counts(index_in_group, start_inclusive,
                                old_open_mins_before_group[group_idx],
                                is_best_edge_min, old_num_per_group[group_idx]);
          new_num_per_group[output_idx] = num_left;
          new_num_per_group[output_idx + 1] = num_right;
          new_open_mins_before_group[output_idx] =
              old_open_mins_before_group[group_idx];
          new_open_mins_before_group[output_idx + 1] = open_mins_before_right;
          new_better_than_no_split[output_idx] =
              old_better_than_no_split[group_idx];
          new_better_than_no_split[output_idx + 1] =
              old_better_than_no_split[group_idx];
        } else {
          // z group sizes
          unsigned start =previous_values[1];
          unsigned size = group_size(group_idx, old_groups_arr[1]);
          nodes[node_offset + group_idx] = DirTreeNode(start, start + size);
        }
      });

  node_offset_ = new_node_offset;

  std::swap(axis_groups_.first, axis_groups_.second);
  std::swap(open_mins_before_group_.first, open_mins_before_group_.second);
  std::swap(num_per_group_.first, num_per_group_.second);
  std::swap(better_than_no_split_.first, better_than_no_split_.second);
}

template class DirTreeGeneratorImpl<ExecutionModel::CPU>;
template class DirTreeGeneratorImpl<ExecutionModel::GPU>;
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
