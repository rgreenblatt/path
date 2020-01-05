#include "ray/sort_actions.h"
#include "ray/sort_actions_impl.h"

namespace ray {
namespace detail {
void sort_actions_cpu(Span<const Traversal, false> traversals,
                      Span<Action> actions) {
/* #pragma omp parallel for */
  for (unsigned traversal_idx = 0; traversal_idx < traversals.size();
       traversal_idx++) {
    sort_traversal_actions(traversals[traversal_idx], actions);
  }
}
} // namespace detail
} // namespace ray
