#include "ray/detail/accel/dir_tree/dir_tree_generator_impl.h"

#include <thrust/scan.h>

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
template <ExecutionModel execution_model>
void DirTreeGeneratorImpl<execution_model>::scan_edges() {
  Span<const uint8_t> to_scan = current_edges_->is_mins();
  Span<const unsigned> keys = current_edges_keys_.get();

  starts_inclusive_.resize(current_edges_->size());

  thrust::inclusive_scan_by_key(thrust_data_[0].execution_policy(),
                                keys.begin(),
                                keys.begin() + current_edges_keys_->size(),
                                to_scan.begin(), starts_inclusive_.begin());
}

template class DirTreeGeneratorImpl<ExecutionModel::GPU>;
template class DirTreeGeneratorImpl<ExecutionModel::CPU>;
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
