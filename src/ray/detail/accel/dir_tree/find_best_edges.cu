#include "lib/async_for.h"
#include "lib/printf_dbg.h"
#include "ray/detail/accel/dir_tree/dir_tree_generator_impl.h"
#include "ray/detail/accel/dir_tree/group.h"
#include "ray/detail/accel/dir_tree/left_right_counts.h"

#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
HOST_DEVICE inline float cost_heuristic(unsigned num_left, unsigned num_right,
                                        float prop_left) {
  // surface area heuristic (traversal cost and intersection cost terms
  // will be used elsewhere)
  return prop_left * num_left + (1 - prop_left) * num_right;
}

HOST_DEVICE inline float get_prop_left(float value, float min_value,
                                       float max_value) {
  return (value - min_value) / (max_value - min_value);
}

template <ExecutionModel execution_model>
void DirTreeGeneratorImpl<execution_model>::find_best_edges() {
  Span<const unsigned> keys = current_edges_keys_.get();
  Span<const unsigned> groups = current_edges_groups();
  Span<const unsigned> starts_inclusive(starts_inclusive_);
  Span<const float> edge_values = current_edges_->values();
  Span<const uint8_t> edge_is_min = current_edges_->is_mins();
  Span<const unsigned> open_mins_before_group =
      open_mins_before_group_.first.get();
  Span<const unsigned> num_per_group = num_per_group_.first.get();

  thrust::reduce_by_key(
      thrust_data_[0].execution_policy(), keys.begin(),
      keys.begin() + current_edges_keys_->size(),
      thrust::make_transform_iterator(
          thrust::make_counting_iterator(0u),
          [=] __host__ __device__(const unsigned i) {
            auto key = keys[i];
            auto [start, end] = group_start_end(key, groups);

            float first_value_in_region = edge_values[start];
            float last_value_in_region = edge_values[end - 1];
            float this_value = edge_values[i];

            assert(last_value_in_region >= first_value_in_region);

            float prop_left = get_prop_left(this_value, first_value_in_region,
                                            last_value_in_region);

            unsigned start_inclusive = starts_inclusive[i];
            unsigned index_in_group = i - start;
            auto [num_left, num_right, unused] = left_right_counts(
                index_in_group, start_inclusive, open_mins_before_group[key],
                edge_is_min[i], num_per_group[key]);

            float cost = cost_heuristic(num_left, num_right, prop_left);

            return BestEdge(cost, i);
          }),
      thrust::make_discard_iterator(), best_edges_.data(),
      [] __host__ __device__(const unsigned first, const unsigned second) {
        return first == second;
      },
      [] __host__ __device__(const BestEdge &first, const BestEdge &second)
          -> const BestEdge & { return std::min(first, second); });
}

template class DirTreeGeneratorImpl<ExecutionModel::GPU>;
template class DirTreeGeneratorImpl<ExecutionModel::CPU>;
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
