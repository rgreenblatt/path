#pragma once

#include "execution_model/thrust_data.h"
#include "intersectable_scene/to_bulk.h"

#include <thrust/transform.h>

namespace intersectable_scene {
template <ExecutionModel exec, intersect::Intersectable I>
Span<const typename ToBulkGen<exec, I>::IntersectionOp>
ToBulkGen<exec, I>::get_intersections(SpanSized<intersect::Ray> rays) {
  intersections_.resize(rays.size());
  thrust::transform(
      ThrustData<exec>().execution_policy(), rays.begin(), rays.end(),
      intersections_.begin(),
      [intersectable = scene_intersectable_->intersectable](
          const intersect::Ray &ray) { return intersectable.intersect(ray); });

  return intersections_;
}
} // namespace intersectable_scene
