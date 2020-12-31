#include "intersect/accel/kdtree/generator.h"
#include "intersect/accel/kdtree/kdtree.h"

namespace intersect {
namespace accel {
namespace kdtree {
template <ExecutionModel execution_model> KDTree<execution_model>::KDTree() {
  gen_ = std::make_unique<Generator>();
}

template <ExecutionModel execution_model>
KDTree<execution_model>::~KDTree() = default;

template <ExecutionModel execution_model>
KDTree<execution_model>::KDTree(KDTree &&) = default;

template <ExecutionModel execution_model>
KDTree<execution_model> &
KDTree<execution_model>::operator=(KDTree &&) = default;

template <ExecutionModel execution_model>
detail::Ref KDTree<execution_model>::gen_internal(const Settings &settings) {
  auto [nodes, permutation] = gen_->gen(settings, bounds_);

  return detail::Ref{nodes, permutation};
}

template class KDTree<ExecutionModel::CPU>;
template class KDTree<ExecutionModel::GPU>;
} // namespace kdtree
} // namespace accel
} // namespace intersect
