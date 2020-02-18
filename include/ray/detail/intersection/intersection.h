#pragma once

#include "lib/cuda/utils.h"

#include <Eigen/Core>

#include <compare>

namespace ray {
namespace detail {
using UVPosition = Eigen::Array2f;

struct IntersectionNormalUV {
  float intersection;
  Eigen::Vector3f normal;
  UVPosition uv;

  HOST_DEVICE IntersectionNormalUV(float intersection,
                                   const Eigen::Vector3f &normal,
                                   const UVPosition &uv)
      : intersection(intersection), normal(normal), uv(uv) {}

  constexpr std::partial_ordering
  operator<=>(const IntersectionNormalUV &rhs) const {
    return intersection <=> rhs.intersection;
  }
};

template <bool normal_and_uv>
using Intersection =
    typename std::conditional<normal_and_uv, IntersectionNormalUV, float>::type;

template <bool normals_and_uv> struct BestIntersectionGeneral {
  using Intersection = Intersection<normals_and_uv>;
  Intersection intersection;
  unsigned shape_idx;

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
