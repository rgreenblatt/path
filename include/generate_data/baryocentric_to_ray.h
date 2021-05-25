#pragma once

#include "integrate/sample_triangle.h"
#include "intersect/ray.h"
#include "intersect/triangle.h"
#include "lib/attribute.h"

namespace generate_data {
ATTR_PURE_NDEBUG inline intersect::Ray
baryocentric_to_ray(float s, float t, const intersect::Triangle &triangle,
                    const UnitVector &dir) {
  auto point = triangle.baryo_to_point({s, t});

  return {
      .origin = point - (*dir) * 1e-5,
      .direction = dir,
  };
}
} // namespace generate_data
