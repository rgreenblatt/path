#include "ray/detail/accel/dir_tree/dir_tree_generator.h"

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
template <ExecutionModel execution_model>
void DirTreeGenerator<execution_model>::construct_trees() {
  // approach:
  // - compute scan for ends and starts
  // - test all edge choices and reduce to best choice (test both x and y for
  //   now, later test alternating).
  // - using best choice, write out where new division will be
  // - filter the other edges using a prefix sum etc
  // - filter sorted by z min and z max using a prefix sum etc

  // generalized segmented prefix sum is important...
  // generalized segmented transform is important...
  // look at how thrust does things and see if special casing is important
  // look at using bit fields or uint8_t to store filter condition

  // 1. approach to segmented ___:
  //  - index globally
  //  - operate from there
  //  - fill global index using dynamic kernel launches where needed and
  //   otherwise looping...

  // 2. approach to segmented ___:
  //  - index into thread block
  //  - either entire thread block or start of data per warp
  //    index into warp
  //  - either entire warp or start of data per index

}
template class DirTreeGenerator<ExecutionModel::CPU>;
template class DirTreeGenerator<ExecutionModel::GPU>;
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
