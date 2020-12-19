#include "intersect/accel/kdtree/kdtree.h"
#include "intersect/accel/kdtree/generator.h"

namespace intersect {
namespace accel {
namespace kdtree {
template <ExecutionModel execution_model>
KDTree<execution_model>::KDTree() {
  gen_ = std::make_unique<Generator>();
}

// TODO (needed for unique_ptr...)
template <ExecutionModel execution_model>
KDTree<execution_model>::~KDTree() = default;

template <ExecutionModel execution_model>
KDTree<execution_model>::KDTree(KDTree &&) = default;

template <ExecutionModel execution_model>
KDTree<execution_model> &
KDTree<execution_model>::operator=(KDTree &&) = default;

template class KDTree<ExecutionModel::CPU>;
template class KDTree<ExecutionModel::GPU>;
} // namespace kdtree
} // namespace accel
} // namespace intersect
