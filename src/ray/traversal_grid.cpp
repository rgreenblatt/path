#include "ray/projection_impl.h"
#include "ray/traversal_grid_impl.h"

namespace ray {
namespace detail {
void update_shapes_cpu(Span<TraversalGrid, false> grids,
                       Span<ShapePossibles> shape_grids,
                       Span<const BoundingPoints> shape_bounds,
                       unsigned num_shapes) {
#pragma omp parallel for
  for (unsigned i = 0; i < grids.size(); i++) {
    for (unsigned shape = 0; shape < num_shapes; shape++) {
      grids[i].updateShape(shape_bounds, shape_grids, shape);
    }
  }
}

void update_counts_cpu(Span<TraversalGrid, false> grids,
                       Span<const ShapePossibles> shape_grids, Span<int> counts,
                       unsigned num_shapes) {
#pragma omp parallel for
  for (unsigned i = 0; i < grids.size(); i++) {
    for (unsigned shape = 0; shape < num_shapes; shape++) {
      grids[i].getCount(shape_grids, shape, counts);
    }
  }
}

void add_actions_cpu(Span<TraversalGrid, false> grids,
                     Span<const ShapePossibles> shape_grids,
                     Span<int> action_indexes, Span<Action> actions,
                     unsigned num_shapes) {
#pragma omp parallel for
  for (unsigned i = 0; i < grids.size(); i++) {
    for (unsigned shape = 0; shape < num_shapes; shape++) {
      grids[i].addActions(shape_grids, shape, action_indexes, actions);
    }
  }
}
} // namespace detail
} // namespace ray
