#pragma once

#include "intersect/accel/aabb.h"
#include "intersect/triangle.h"

namespace intersect {
namespace accel {
namespace detail {
struct ClippedTriangle {
  ClippedTriangle() = default;

  HOST_DEVICE inline ClippedTriangle(const Triangle &triangle)
      : bounding(triangle.bounds()), triangle(triangle) {}

  AABB bounding;
  Triangle triangle;

  HOST_DEVICE inline AABB new_bounds(const float left_bound,
                                     const float right_bound,
                                     const unsigned axis) const;
};
} // namespace detail
} // namespace accel
} // namespace intersect
