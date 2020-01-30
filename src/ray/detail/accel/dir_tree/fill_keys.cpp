#include "lib/span_convertable_vector.h"
#include "ray/detail/accel/dir_tree/dir_tree_generator.h"

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
template <> void DirTreeGenerator<ExecutionModel::CPU>::fill_keys() {
  std::array<SpanSized<unsigned>, 3> keys = {x_edges_keys_, y_edges_keys_,
                                             z_keys_};

#pragma omp parallel for collapse(2) schedule(dynamic, 16)
  for (unsigned axis = 0; axis < 3; axis++) {
    for (unsigned i = 0; i < divisions_.size(); i++) {
      std::fill(keys[axis].begin() + divisions_[i].starts[axis],
                keys[axis].begin() + divisions_[i].ends[axis], i);
    }
  }
}
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray