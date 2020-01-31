#include "ray/detail/accel/dir_tree/dir_tree_generator.h"

#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>

#include "lib/async_for.h"

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {

float cost_heuristic(unsigned num_left, unsigned num_right, float prop_left) {
  // surface area heuristic (traversal cost and intersection cost terms
  // will be used elsewhere)
  return prop_left * num_left + (1 - prop_left) * num_right;
}

template <ExecutionModel execution_model>
void DirTreeGenerator<execution_model>::find_best_edges(bool is_x) {
  Span<const unsigned> keys = is_x ? x_edges_keys_ : y_edges_keys_;
  thrust::reduce_by_key(
      thrust_data_[0].execution_policy(), keys.begin(), keys.end(),
      thrust::make_transform_iterator(
          thrust::make_counting_iterator(0u),
          [=] __host__ __device__(const unsigned i) { 
          return BestEdge(); 
          }),
      thrust::make_discard_iterator(), best_edges_.begin(),
      [] __host__ __device__(const unsigned first, const unsigned second) {
        return first == second;
      },
      [] __host__ __device__(const BestEdge &first, const BestEdge &second)
          -> const BestEdge & { return std::min(first, second); });
}

template class DirTreeGenerator<ExecutionModel::GPU>;
template class DirTreeGenerator<ExecutionModel::CPU>;
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
