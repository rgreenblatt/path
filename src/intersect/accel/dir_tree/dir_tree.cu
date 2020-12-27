#include "intersect/accel/dir_tree/dir_tree.h"
#include "lib/assert.h"

namespace intersect {
namespace accel {
namespace dir_tree {
template <ExecutionModel execution_model>
template <Bounded B>
typename detail::Ref DirTree<execution_model>::gen(const Settings &,
                                                   SpanSized<const B>,
                                                   const AABB &aabb) {
  // TODO
  unreachable();
  return detail::Ref{aabb};
}

template class DirTree<ExecutionModel::CPU>;
template class DirTree<ExecutionModel::GPU>;
} // namespace dir_tree
} // namespace accel
} // namespace intersect
