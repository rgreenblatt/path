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
  Span<const unsigned> num_per_group = num_per_group_.first.get();
  Span<const std::array<float, 2>> group_min_max = current_edges_min_max_.get();

  best_edges_.resize_all(num_groups());
  best_edges_locations_.resize(num_groups());

  auto start_it = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0u),
      [=] __host__ __device__(const unsigned group_idx) {
        return group_size(group_idx, groups) > 0;
      });

  // SPEED: could be executed at same time as below...
  thrust::inclusive_scan(thrust_data_[0].execution_policy(), start_it,
                         start_it + num_groups(), best_edges_locations_.data());

  using BestEdge = thrust::tuple<float, unsigned, uint8_t>;

  thrust::reduce_by_key(
      thrust_data_[0].execution_policy(), keys.begin(),
      keys.begin() + current_edges_keys_->size(),
      thrust::make_transform_iterator(
          thrust::make_counting_iterator(0u),
          [=] __host__ __device__(const unsigned i) -> BestEdge {
            auto key = keys[i];
            auto [start, end] = group_start_end(key, groups);

            std::array<float, 2> min_max_region = group_min_max[key];
            float this_value = edge_values[i];

            assert(i >= start);
            assert(i < end);
            assert(min_max_region[0] <= min_max_region[1]);
            assert(min_max_region[0] <= this_value + 1e-5);
            assert(this_value <= min_max_region[1] + 1e-5);

            float prop_left =
                get_prop_left(this_value, min_max_region[0], min_max_region[1]);

            unsigned start_inclusive = starts_inclusive[i];
            unsigned index_in_group = i - start;
            auto [num_left, num_right] = left_right_counts(
                index_in_group, start_inclusive, starts_inclusive[end - 1],
                edge_is_min[i], num_per_group[key]);

            float cost = cost_heuristic(num_left, num_right, prop_left);

            return {cost, i, num_left == 0 || num_right == 0};
          }),
      thrust::make_discard_iterator(),
      thrust::make_zip_iterator(thrust::make_tuple(
          best_edges_.costs().data(), best_edges_.idxs().data(),
          best_edges_.side_of_size_zero().data())),
      [] __host__ __device__(const unsigned first, const unsigned second) {
        return first == second;
      },
      [] __host__ __device__(const BestEdge &first,
                             const BestEdge &second) -> const BestEdge & {
        if (thrust::get<0>(first) < thrust::get<0>(second)) {
          return first;
        } else {
          return second;
        }
      });
}

template class DirTreeGeneratorImpl<ExecutionModel::GPU>;
template class DirTreeGeneratorImpl<ExecutionModel::CPU>;
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
