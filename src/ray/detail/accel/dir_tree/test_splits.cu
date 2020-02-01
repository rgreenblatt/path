#include "lib/cuda/utils.h"
#include "lib/span_convertable_device_vector.h"
#include "lib/span_convertable_vector.h"
#include "ray/detail/accel/dir_tree/dir_tree_generator_impl.h"

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
HOST_DEVICE inline bool use_division(const BestEdge &best_edge,
                                     unsigned group_size) {
  const float traversal_cost = 1;
  const float intersection_cost = 1;

  float division_cost = best_edge.cost * intersection_cost + traversal_cost;
  float no_division_cost = group_size * intersection_cost;

  return division_cost < no_division_cost;
}

template <ExecutionModel execution_model>
void DirTreeGeneratorImpl<execution_model>::test_splits() {
  Span<const BestEdge> best_edges = best_edges_;
  Span<const unsigned> num_per_group = num_per_group_;
  auto start_counting_it = thrust::make_counting_iterator(0u);
  auto end_counting_it = start_counting_it + num_groups();
  thrust::transform(start_counting_it, end_counting_it,
                    better_than_no_split_.first->begin(),
                    [=] __host__ __device__(unsigned i) {
                      return use_division(best_edges[i], num_per_group[i]);
                    });
}
template class DirTreeGeneratorImpl<ExecutionModel::CPU>;
template class DirTreeGeneratorImpl<ExecutionModel::GPU>;
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
