#include "lib/cuda/utils.h"
#include "lib/span_convertable_device_vector.h"
#include "lib/span_convertable_vector.h"
#include "ray/detail/accel/dir_tree/dir_tree_generator_impl.h"
#include "ray/detail/accel/dir_tree/group.h"

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
HOST_DEVICE inline bool use_division(float best_edge_cost,
                                     unsigned group_size) {
  const float traversal_cost = 1;
  const float intersection_cost = 1;

  float division_cost = best_edge_cost * intersection_cost + traversal_cost;
  float no_division_cost = group_size * intersection_cost;

  return division_cost < no_division_cost;
}

template <ExecutionModel execution_model>
void DirTreeGeneratorImpl<execution_model>::test_splits() {
  Span<const float> best_edges_cost = best_edges_.costs();
  Span<const unsigned> num_per_group = num_per_group_.first.get();
  auto start_counting_it = thrust::make_counting_iterator(0u);
  auto end_counting_it = start_counting_it + num_groups();
  thrust::transform(start_counting_it, end_counting_it,
                    better_than_no_split_.first->begin(),
                    [=] __host__ __device__(unsigned i) {
                      return use_division(best_edges_cost[i], num_per_group[i]);
                    });

  Span<const uint8_t> use_split_first = better_than_no_split_.first.get();
  Span<const uint8_t> use_split_second = better_than_no_split_.second.get();
  Span<const unsigned> z_groups = axis_groups_.first.get()[2];
  Span<const uint8_t> best_edge_one_split = best_edges_.side_of_size_zero();
  auto start_it = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0u),

      [=] __host__ __device__(unsigned i) {
        bool using_split = use_split_first[i] || use_split_second[i];
        unsigned z_output_size = using_split ? 0 : group_size(i, z_groups);

        unsigned output_split_groups =
            using_split ? 2 - best_edge_one_split[i] : 0;

        // if one side will have size zero, don't include that side
        return thrust::make_tuple(output_split_groups, z_output_size);
      });

  unsigned size = better_than_no_split_.first->size();

  num_groups_inclusive_.resize(size);
  z_outputs_inclusive_.resize(size);

  thrust::inclusive_scan(
      thrust_data_[0].execution_policy(), start_it, start_it + size,
      thrust::make_zip_iterator(thrust::make_tuple(
          num_groups_inclusive_.begin(), z_outputs_inclusive_.begin())),
      [] __host__ __device__(const thrust::tuple<unsigned, unsigned> &l,
                             const thrust::tuple<unsigned, unsigned> &r) {
        return thrust::make_tuple(thrust::get<0>(l) + thrust::get<0>(r),
                                  thrust::get<1>(l) + thrust::get<1>(r));
      });
}

template class DirTreeGeneratorImpl<ExecutionModel::CPU>;
template class DirTreeGeneratorImpl<ExecutionModel::GPU>;
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
