#include "lib/async_for.h"
#include "ray/detail/accel/dir_tree/dir_tree_generator.h"

#include <thrust/sort.h>

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
template <ExecutionModel execution_model>
void DirTreeGenerator<execution_model>::sort() {
  async_for<true>(0, num_sortings, [&](unsigned i) {
    thrust::sort_by_key(thrust_data_[i].execution_policy(),
                        sorting_values_[i].data(),
                        sorting_values_[i].data() + sorting_values_[i].size(),
                        indexes_[i].data());
  });
}

template class DirTreeGenerator<ExecutionModel::CPU>;
template class DirTreeGenerator<ExecutionModel::GPU>;
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
