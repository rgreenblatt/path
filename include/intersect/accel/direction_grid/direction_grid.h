#pragma once

#include "execution_model/execution_model_vector_type.h"
#include "execution_model/thrust_data.h"
#include "intersect/accel/detail/bvh/bvh.h"
#include "intersect/accel/direction_grid/settings.h"
#include "intersect/accel/triangle_accel.h"
#include "intersect/triangle.h"
#include "lib/attribute.h"
#include "meta/all_values/impl/enum.h"

#include <memory>

namespace intersect {
namespace accel {
namespace direction_grid {
namespace detail {
// doesn't depend on execution_model
class DirectionGridRef {
public:
  template <IntersectableAtIdx F>
  HOST_DEVICE inline AccelRet<F>
  intersect_objects(const intersect::Ray &ray,
                    const F &intersectable_at_idx) const;

  struct IntersectionIdxs {
    unsigned face;
    unsigned i;
    unsigned j;
  };

  HOST_DEVICE inline static unsigned
  idx(const std::array<IntersectionIdxs, 2> &intersections, unsigned grid);

  Span<const unsigned> overall_idxs;
  Span<const StartEnd<unsigned>> direction_idxs;
  Span<const AABB> node_bounds;
  Span<const unsigned> node_grid;
};
} // namespace detail

template <ExecutionModel exec> class DirectionGrid {
public:
  // need to implementated when Generator is defined
  DirectionGrid();
  ~DirectionGrid();
  DirectionGrid(DirectionGrid &&);
  DirectionGrid &operator=(DirectionGrid &&);

  using Ref = detail::DirectionGridRef;

  RefPerm<Ref> gen(const Settings &settings,
                   SpanSized<const Triangle> triangles);

private:
  // PIMPL
  class Generator;

  std::unique_ptr<Generator> gen_;
};

// NOTE: this could be more general, but it doesn't need to be right now.
// We would need some notion in which objects could be "clipped".
template <ExecutionModel exec>
struct IsTriangleAccel
    : std::bool_constant<TriangleAccel<DirectionGrid<exec>, Settings>> {};

static_assert(PredicateForAllValues<ExecutionModel>::value<IsTriangleAccel>);
} // namespace direction_grid
} // namespace accel
} // namespace intersect
