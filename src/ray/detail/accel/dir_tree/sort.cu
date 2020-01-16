#include "ray/detail/accel/dir_tree/dir_tree_generator.h"

#include <thrust/sort.h>

#include <future>

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
template <ExecutionModel execution_model>
void DirTreeGenerator<execution_model>::sort() {
  std::array<std::future<void>, num_sortings> results;

  for (unsigned i = 0; i < num_sortings; i++) {
    results[i] = std::async(std::launch::async, [&]() {
      thrust::sort_by_key(thrust_data_[i].execution_policy(),
                          sorting_values_[i].data(),
                          sorting_values_[i].data() + sorting_values_[i].size(),
                          indexes_[i].data());
    });
  }

  for (auto &result : results) {
    result.get();
  }
}

template class DirTreeGenerator<ExecutionModel::CPU>;
template class DirTreeGenerator<ExecutionModel::GPU>;
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
