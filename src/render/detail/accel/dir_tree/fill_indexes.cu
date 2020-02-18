#include "lib/async_for.h"
#include "ray/detail/accel/dir_tree/dir_tree_generator_impl.h"

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
template <ExecutionModel execution_model>
void DirTreeGeneratorImpl<execution_model>::fill_indexes() {
  async_for(use_async_, 0, num_sortings, [&](unsigned i) {
    thrust::copy(thrust_data_[i].execution_policy(),
                 thrust::make_counting_iterator(0u),
                 thrust::make_counting_iterator(unsigned(indexes_[i].size())),
                 indexes_[i].data());
  });
}

template class DirTreeGeneratorImpl<ExecutionModel::GPU>;
template class DirTreeGeneratorImpl<ExecutionModel::CPU>;
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
