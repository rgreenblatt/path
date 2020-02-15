#include "ray/detail/accel/dir_tree/dir_tree_generator.h"
#include "ray/detail/accel/dir_tree/dir_tree_generator_impl.h"

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
template <ExecutionModel execution_model>
DirTreeGenerator<execution_model>::DirTreeGenerator()
    : ptr_(new DirTreeGeneratorImpl<execution_model>()) {}

template <ExecutionModel execution_model>
DirTreeGenerator<execution_model>::~DirTreeGenerator() {
  delete ptr_;
}

template <ExecutionModel execution_model>
DirTreeLookup DirTreeGenerator<execution_model>::generate(
    SpanSized<const scene::ShapeData> shapes, unsigned target_num_dir_trees,
    const Eigen::Vector3f &min_bound, const Eigen::Vector3f &max_bound) {
  return ptr_->generate(shapes, target_num_dir_trees, min_bound, max_bound);
}

template class DirTreeGenerator<ExecutionModel::GPU>;
template class DirTreeGenerator<ExecutionModel::CPU>;
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
