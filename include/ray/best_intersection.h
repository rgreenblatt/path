#pragma once

#include "lib/cuda_utils.h"
#include "scene/scene.h"

namespace ray {
namespace detail {
struct BestIntersection {
  float intersection;
  unsigned shape_index;
  scene::Shape shape_type;

  __host__ __device__ BestIntersection(const float intersection,
                                       const unsigned shape_index,
                                       const scene::Shape shape_type)
      : intersection(intersection), shape_index(shape_index),
        shape_type(shape_type) {}

  __host__ __device__ bool operator<(const BestIntersection &rhs) const {
    return intersection < rhs.intersection;
  }
};
} // namespace detail
} // namespace ray
