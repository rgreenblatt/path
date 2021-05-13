#pragma once

#include "intersect/triangle.h"

namespace intersect {
// TODO: more efficient general approach along with is_triangle_between
HOST_DEVICE inline bool
is_triangle_blocking(const std::array<Triangle, 2> &tris,
                     const Triangle &possibly_blocking) {
  for (unsigned i = 0; i < 3; ++i) {
    for (unsigned j = 0; j < 3; ++j) {
      auto origin = tris[0].vertices[i];
      auto dir = tris[1].vertices[j] - origin;
      float distance = dir.norm();

      auto intersection = possibly_blocking.intersect(intersect::Ray{
          .origin = origin,
          .direction = UnitVector::new_normalize(dir),
      });
      if (!intersection.has_value() || intersection->intersection_dist < 0.f ||
          intersection->intersection_dist > distance) {
        return false;
      }
    }
  }

  return true;
}
} // namespace intersect
