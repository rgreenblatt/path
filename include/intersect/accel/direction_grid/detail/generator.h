#pragma once

#include "execution_model/execution_model_vector_type.h"
#include "intersect/accel/aabb.h"
#include "intersect/accel/direction_grid/direction_grid.h"
#include "intersect/accel/direction_grid/settings.h"
#include "lib/span.h"

#include <tuple>

namespace intersect {
namespace accel {
namespace direction_grid {
template <ExecutionModel exec> class DirectionGrid<exec>::Generator {
public:
  Generator() = default;

  RefPerm<Ref> gen(const Settings &settings,
                   SpanSized<const Triangle> triangles);

private:
  ExecVector<exec, unsigned> overall_idxs_;
  ExecVector<exec, StartEnd<unsigned>> direction_idxs_;
  ExecVector<exec, AABB> node_bounds_;
  ExecVector<exec, unsigned> node_grid_;
};
} // namespace direction_grid
} // namespace accel
} // namespace intersect
