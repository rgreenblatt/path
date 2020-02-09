#include "lib/async_for.h"
#include "lib/cuda/utils.h"
#include "lib/span_convertable_device_vector.h"
#include "lib/span_convertable_vector.h"
#include "ray/detail/accel/dir_tree/dir_tree_generator_impl.h"
#include "ray/detail/accel/dir_tree/group.h"

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
template <typename FResize, typename FCopyTo, typename FCopyToOutputs,
          typename ExecPolicy>
void filter_values(
    Span<const float> edge_values, Span<const BestEdge> best_edges,
    Span<const float> compare_data_mins, Span<const float> compare_data_maxs,
    Span<const unsigned> keys, Span<const unsigned> groups,
    Span<const uint8_t> use_split_first, Span<const uint8_t> use_split_second,
    Span<unsigned> new_indexes, unsigned size, const FResize &resize,
    const FCopyTo &copy_to_new, const FCopyToOutputs &copy_to_outputs,
    const ExecPolicy &execution_policy) {
  auto get_key_idx_is_left = [=] __host__ __device__(unsigned i) {
    unsigned key = keys[i / 2];
    auto [start, end] = group_start_end(key, groups);
    unsigned size = group_size(key, groups);
    unsigned within_group_idx = i - start * 2;
    bool is_left = within_group_idx < size;
    if (!is_left) {
      within_group_idx -= size;
    }
    unsigned idx = within_group_idx + start;

    return std::make_tuple(key, idx, is_left);
  };

  auto start_it = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0u), [=] __host__ __device__(unsigned i) {
        auto [key, idx, is_left] = get_key_idx_is_left(i);

        if (!(use_split_first[key] || use_split_second[key])) {
          return false;
        }

        float edge_value = edge_values[best_edges[key].idx];

        return is_left ? compare_data_mins[idx] < edge_value
                       : compare_data_maxs[idx] > edge_value;
      });

  thrust::inclusive_scan(execution_policy, start_it, start_it + size,
                         new_indexes.begin());

  resize();

  auto start_counting_it = thrust::make_counting_iterator(0u);
  thrust::for_each(execution_policy, start_counting_it,
                   start_counting_it + size,
                   [=] __host__ __device__(unsigned i) {
                     unsigned previous_value = get_previous(i, new_indexes);
                     auto [key, idx, is_left] = get_key_idx_is_left(i);
                     if (new_indexes[i] != previous_value) {
                       copy_to_new(idx, previous_value);
                     } else {
                       copy_to_outputs(idx, key);
                     }
                   });
}

template <ExecutionModel execution_model>
void DirTreeGeneratorImpl<execution_model>::filter_others() {
  Span<const BestEdge> best_edges = best_edges_;
  Span<const float> edge_values = current_edges_->values();

  unsigned new_size = output_values_offset_ +
                      z_outputs_inclusive_[z_outputs_inclusive_.size() - 1];

  output_keys_.resize(new_size);
  min_sorted_values_.resize(new_size);
  min_sorted_inclusive_maxes_.resize(new_size);
  min_sorted_indexes_.resize(new_size);
  max_sorted_values_.resize(new_size);
  max_sorted_inclusive_mins_.resize(new_size);
  max_sorted_indexes_.resize(new_size);

  async_for<true>(0, 3, [&](unsigned i) {
    if (i == 0) {
      unsigned size = other_edges_->size() * 2;

      new_edge_indexes_.resize(size);

      Span<const float> compare_data_mins = other_edges_->other_mins();
      Span<const float> compare_data_maxs = other_edges_->other_maxs();
      Span<const unsigned> keys = other_edges_keys_.get();
      Span<const unsigned> groups = other_edges_groups();

      Span<const float> other_edges_other_mins = other_edges_->other_mins();
      Span<const float> other_edges_other_maxs = other_edges_->other_maxs();
      Span<const float> other_edges_values = other_edges_->values();
      Span<const uint8_t> other_edges_is_mins = other_edges_->is_mins();

      Span<float> new_other_edges_other_mins;
      Span<float> new_other_edges_other_maxs;
      Span<float> new_other_edges_values;
      Span<uint8_t> new_other_edges_is_mins;

      filter_values(
          edge_values, best_edges, compare_data_mins, compare_data_maxs, keys,
          groups, better_than_no_split_.first.get(),
          better_than_no_split_.second.get(),

          new_edge_indexes_, size,
          [&] {
            unsigned new_size = new_edge_indexes_[size - 1];
            other_edges_new_->resize_all(new_size);

            new_other_edges_other_mins = other_edges_new_->other_mins();
            new_other_edges_other_maxs = other_edges_new_->other_maxs();
            new_other_edges_values = other_edges_new_->values();
            new_other_edges_is_mins = other_edges_new_->is_mins();
          },
          [=] __host__ __device__(unsigned old_index, unsigned new_index) {
            new_other_edges_other_mins[new_index] =
                other_edges_other_mins[old_index];
            new_other_edges_other_maxs[new_index] =
                other_edges_other_maxs[old_index];
            new_other_edges_values[new_index] = other_edges_values[old_index];
            new_other_edges_is_mins[new_index] = other_edges_is_mins[old_index];
          },
          [] __host__ __device__(unsigned, unsigned) {},
          thrust_data_[i].execution_policy());

    } else {
      Span<const unsigned> keys = z_keys_;

      bool is_x_min = i == 1;

      ZValues &old_z_vals =
          (is_x_min ? sorted_by_z_min_ : sorted_by_z_max_).first;
      ZValues &new_z_vals =
          (is_x_min ? sorted_by_z_min_ : sorted_by_z_max_).second;

      // kinda extra...
      Span<const float> old_z_vals_x_min = old_z_vals.x_mins();
      Span<const float> old_z_vals_x_max = old_z_vals.x_maxs();
      Span<const float> old_z_vals_y_min = old_z_vals.y_mins();
      Span<const float> old_z_vals_y_max = old_z_vals.y_maxs();
      Span<const float> old_z_vals_z_min = old_z_vals.z_mins();
      Span<const float> old_z_vals_z_max = old_z_vals.z_maxs();
      Span<const unsigned> old_z_vals_idx = old_z_vals.idxs();

      Span<float> new_z_vals_x_min;
      Span<float> new_z_vals_x_max;
      Span<float> new_z_vals_y_min;
      Span<float> new_z_vals_y_max;
      Span<float> new_z_vals_z_min;
      Span<float> new_z_vals_z_max;
      Span<unsigned> new_z_vals_idx;

      Span<const float> compare_data_mins =
          is_x_ ? old_z_vals.x_mins() : old_z_vals.y_mins();
      Span<const float> compare_data_maxs =
          is_x_ ? old_z_vals.x_maxs() : old_z_vals.y_maxs();
      Span<const unsigned> groups = axis_groups_.first.get()[2];
      Span<const unsigned> outputs_inclusive = z_outputs_inclusive_;

      auto values = old_z_vals_z_max;
      auto other_values = old_z_vals_z_min;
      if (is_x_min) {
        std::swap(values, other_values);
      }

      Span<float> output_values =
          is_x_min ? min_sorted_values_ : max_sorted_values_;
      Span<float> output_other_values =
          is_x_min ? min_sorted_inclusive_maxes_ : max_sorted_inclusive_mins_;
      Span<unsigned> output_indexes =
          is_x_min ? min_sorted_indexes_ : max_sorted_indexes_;
      Span<unsigned> output_keys = output_keys_;

      unsigned output_offset = output_values_offset_;

      unsigned size = old_z_vals.size() * 2;

      auto &indexes = i == 1 ? new_z_min_indexes_ : new_z_max_indexes_;

      indexes.resize(size);

      filter_values(
          edge_values, best_edges, compare_data_mins, compare_data_maxs, keys,
          groups, better_than_no_split_.first.get(),
          better_than_no_split_.second.get(), indexes, size,
          [&] {
            unsigned new_size = indexes[size - 1];
            new_z_vals.resize_all(new_size);

            new_z_vals_x_min = new_z_vals.x_mins();
            new_z_vals_x_max = new_z_vals.x_maxs();
            new_z_vals_y_min = new_z_vals.y_mins();
            new_z_vals_y_max = new_z_vals.y_maxs();
            new_z_vals_z_min = new_z_vals.z_mins();
            new_z_vals_z_max = new_z_vals.z_maxs();
            new_z_vals_idx = new_z_vals.idxs();
          },
          [=] __host__ __device__(unsigned old_index, unsigned new_index) {
            new_z_vals_x_min[new_index] = old_z_vals_x_min[old_index];
            new_z_vals_x_max[new_index] = old_z_vals_x_max[old_index];
            new_z_vals_y_min[new_index] = old_z_vals_y_min[old_index];
            new_z_vals_y_max[new_index] = old_z_vals_y_max[old_index];
            new_z_vals_z_min[new_index] = old_z_vals_z_min[old_index];
            new_z_vals_z_max[new_index] = old_z_vals_z_max[old_index];
            new_z_vals_idx[new_index] = old_z_vals_idx[old_index];
          },
          [=] __host__ __device__(unsigned idx, unsigned key) {
            unsigned start_idx_for_group =
                get_previous(key, outputs_inclusive) + output_offset;
            auto [start, end] = group_start_end(key, groups);
            unsigned general_idx = start_idx_for_group + (idx - start);

            output_values[general_idx] = values[idx];
            output_other_values[general_idx] = other_values[idx];
            output_indexes[general_idx] = old_z_vals_idx[idx];
            if (is_x_min) {
              // output offset is larger than needed, but should work
              output_keys[general_idx] = key + output_offset;
            }
          },
          thrust_data_[i].execution_policy());
    }
  });
}
template class DirTreeGeneratorImpl<ExecutionModel::CPU>;
template class DirTreeGeneratorImpl<ExecutionModel::GPU>;
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
