#include "ray/detail/accel/dir_tree/dir_tree_generator_impl.h"

#include <thrust/scan.h>

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
template <ExecutionModel execution_model>
void DirTreeGeneratorImpl<execution_model>::scan_mins_maxs() {
  auto zip_it = thrust::make_zip_iterator(thrust::make_tuple(
      min_sorted_inclusive_maxes_.data(), max_sorted_inclusive_mins_.data()));

  using TupleType = thrust::tuple<float, float>;

  thrust::inclusive_scan_by_key(
      output_keys_.data(), output_keys_.data() + output_keys_.size(), zip_it,
      zip_it,
      [] __host__ __device__(const unsigned first, const unsigned second) {
        return first == second;
      },
      [] __host__ __device__(const TupleType &l,
                             const TupleType &r) -> TupleType {
        return {std::max(thrust::get<0>(l), thrust::get<0>(r)),
                std::min(thrust::get<1>(l), thrust::get<1>(r))};
      });
}

template class DirTreeGeneratorImpl<ExecutionModel::GPU>;
template class DirTreeGeneratorImpl<ExecutionModel::CPU>;
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
