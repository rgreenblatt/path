#pragma once

#include "intersectable_scene/to_bulk.h"

#include <thrust/transform.h>

namespace intersectable_scene {
template <ExecutionModel exec, intersect::Intersectable I>
Span<const typename ToBulkGen<exec, I>::IntersectionOp>
ToBulkGen<exec, I>::get_intersections() {
  // TODO: consider if it would be better to use a custom thread division
  // instead of thrust::transform...
  thrust::transform(
      rays_.begin(), rays_.end(), intersections_.begin(),
      [intersectable = scene_intersectable_->intersectable](
          const intersect::Ray &ray) { return intersectable.intersect(ray); });

  return intersections_;
}
} // namespace intersectable_scene
