#include "intersect/accel/direction_grid/detail/generator.h"
#include "intersect/accel/direction_grid/direction_grid.h"

namespace intersect {
namespace accel {
namespace direction_grid {
template <ExecutionModel exec> DirectionGrid<exec>::DirectionGrid() {
  gen_ = std::make_unique<Generator>();
}

template <ExecutionModel exec> DirectionGrid<exec>::~DirectionGrid() = default;

template <ExecutionModel exec>
DirectionGrid<exec>::DirectionGrid(DirectionGrid &&) = default;

template <ExecutionModel exec>
DirectionGrid<exec> &DirectionGrid<exec>::operator=(DirectionGrid &&) = default;

template <ExecutionModel exec>
RefPerm<typename DirectionGrid<exec>::Ref>
DirectionGrid<exec>::gen(const Settings &settings,
                         SpanSized<const Triangle> triangles) {
  return gen_->gen(settings, triangles);
}

template class DirectionGrid<ExecutionModel::CPU>;
#ifndef CPU_ONLY
template class DirectionGrid<ExecutionModel::GPU>;
#endif
} // namespace direction_grid
} // namespace accel
} // namespace intersect
