#include "ray/detail/accel/dir_tree/dir_tree_generator.h"

#include <future>

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
template <ExecutionModel execution_model>
void DirTreeGenerator<execution_model>::fill_indexes() {
  std::array<std::future<void>, num_sortings> results;

  for (unsigned i = 0; i < num_sortings; i++) {
    results[i] = std::async(std::launch::async, [&]() {
      thrust::copy(thrust_data_[i].execution_policy(),
                   thrust::make_counting_iterator(0u),
                   thrust::make_counting_iterator(unsigned(indexes_[i].size())),
                   indexes_[i].data());
    });
  }

  for (auto &result : results) {
    result.get();
  }
}

template class DirTreeGenerator<ExecutionModel::GPU>;
template class DirTreeGenerator<ExecutionModel::CPU>;
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
