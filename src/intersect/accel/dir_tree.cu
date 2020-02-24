#include "intersect/accel/dir_tree.h"
#include "intersect/impl/triangle_impl.h"
#include "intersect/transformed_object.h"

namespace intersect {
namespace accel {
template <ExecutionModel execution_model, Object O>
typename AccelImpl<AccelType::DirTree, execution_model, O>::Ref
AccelImpl<AccelType::DirTree, execution_model, O>::gen(
    const AccelSettings<AccelType::DirTree> &, Span<const O>, unsigned,
    unsigned, const AABB &aabb) {
  // TODO
  assert(false);
  return Ref(aabb);
}

template struct AccelImpl<AccelType::DirTree, ExecutionModel::CPU,
                          intersect::Triangle>;
template struct AccelImpl<AccelType::DirTree, ExecutionModel::CPU,
                          intersect::TransformedObject>;
template struct AccelImpl<AccelType::DirTree, ExecutionModel::GPU,
                          intersect::Triangle>;
template struct AccelImpl<AccelType::DirTree, ExecutionModel::GPU,
                          intersect::TransformedObject>;
} // namespace accel
} // namespace intersect
