#include "intersect/accel/dir_tree/dir_tree.h"
#include "lib/assert.h"

namespace intersect {
namespace accel {
namespace dir_tree {
// TODO
template class DirTree<ExecutionModel::CPU>;
template class DirTree<ExecutionModel::GPU>;
} // namespace dir_tree
} // namespace accel
} // namespace intersect
