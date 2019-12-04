#pragma once

#include "lib/cuda_utils.h"
#include "ray/ray_utils.h"
#include "scene/scene.h"

namespace ray {
namespace detail {

template <bool normals_and_uv> struct BestIntersectionGeneral {
  using Intersection = Intersection<normals_and_uv>;
  Intersection intersection;
  uint16_t shape_idx;

  HOST_DEVICE BestIntersectionGeneral(const Intersection &intersection,
                                      const unsigned shape_idx)
      : intersection(intersection), shape_idx(shape_idx) {}

  HOST_DEVICE bool operator<(const BestIntersectionGeneral &rhs) const {
    return intersection < rhs.intersection;
  }

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

using BestIntersectionNormalUV = BestIntersectionGeneral<true>;
using BestIntersection = BestIntersectionGeneral<false>;
} // namespace detail
} // namespace ray
