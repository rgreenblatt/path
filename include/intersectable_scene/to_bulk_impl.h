#pragma once

#include "intersectable_scene/to_bulk.h"

#include <thrust/transform.h>

namespace intersectable_scene {
template <ExecutionModel exec, intersect::Intersectable I>
Span<const typename ToBulkGen<exec, I>::IntersectionOp>
ToBulkGen<exec, I>::get_intersections() {
  thrust::transform(
      rays_.begin(), rays_.end(), intersections_.begin(),
      [intersectable = scene_intersectable_->intersectable](
          const intersect::Ray &ray) { intersectable.intersect(ray); });

  return intersections_;
}
} // namespace intersectable_scene
