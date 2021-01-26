#pragma once

#include "intersectable_scene/to_bulk.h"

#include <thrust/transform.h>

namespace intersectable_scene {
template <ExecutionModel exec, intersect::Intersectable I>
template <typename T>
Span<const typename ToBulkGen<exec, I>::IntersectionOp>
ToBulkGen<exec, I>::get_intersections(SpanSized<T>) {
  unreachable();
  // TODO
  // TODO: consider if it would be better to use a custom thread division
  // instead of thrust::transform...
#if 0
  thrust::transform(
      rays_.begin(), rays_.end(), intersections_.begin(),
      [intersectable = scene_intersectable_->intersectable](
          const intersect::Ray &ray) { return intersectable.intersect(ray); });

  return intersections_;
#endif
}
} // namespace intersectable_scene
