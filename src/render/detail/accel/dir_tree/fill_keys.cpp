#include "lib/span_convertable_vector.h"
#include "ray/detail/accel/dir_tree/dir_tree_generator_impl.h"
#include "ray/detail/accel/dir_tree/group.h"

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
template <> void DirTreeGeneratorImpl<ExecutionModel::CPU>::fill_keys() {
  std::array<SpanSized<unsigned>, 3> keys = {x_edges_keys_, y_edges_keys_,
                                             z_keys_};

  /* #pragma omp parallel for collapse(2) schedule(dynamic, 16) */
  for (unsigned axis = 0; axis < 3; axis++) {
    for (unsigned i = 0; i < num_groups(); i++) {
      auto [start, end] = group_start_end(i, axis_groups_.first.get()[axis]);
      assert(keys[axis].size() >= end);
      std::fill(keys[axis].begin() + start, keys[axis].begin() + end, i);
    }
  }
}
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
