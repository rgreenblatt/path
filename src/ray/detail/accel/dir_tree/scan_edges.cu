#include "ray/detail/accel/dir_tree/dir_tree_generator.h"

#include <thrust/scan.h>

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
template <ExecutionModel execution_model>
void DirTreeGenerator<execution_model>::scan_edges(bool is_x) {
  auto to_scan = is_x ? x_edges_.template get<3>() : y_edges_.template get<3>();
  Span<const unsigned> keys = is_x ? x_edges_keys_ : y_edges_keys_;
  thrust::inclusive_scan_by_key(thrust_data_[0].execution_policy(),
                                keys.begin(), keys.end(), to_scan.begin(),
                                starts_inclusive_.begin());
}

template class DirTreeGenerator<ExecutionModel::GPU>;
template class DirTreeGenerator<ExecutionModel::CPU>;
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
