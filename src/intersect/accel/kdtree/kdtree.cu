#include "intersect/accel/kdtree/generator.h"
#include "intersect/accel/kdtree/kdtree.h"

#include "lib/info/debug_print.h"
#include <magic_enum.hpp>

namespace intersect {
namespace accel {
namespace kdtree {
template <ExecutionModel execution_model> KDTree<execution_model>::KDTree() {
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

template <ExecutionModel execution_model>
detail::Ref KDTree<execution_model>::gen_internal(const Settings &settings,
                                                  const AABB &aabb) {
  auto [nodes, permutation] = gen_->gen(settings, bounds_);

  dbg(magic_enum::enum_name(execution_model));

  return detail::Ref(nodes, permutation, aabb);
}

template class KDTree<ExecutionModel::CPU>;
template class KDTree<ExecutionModel::GPU>;
} // namespace kdtree
} // namespace accel
} // namespace intersect
