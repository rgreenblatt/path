#pragma once

#include "lib/cuda/utils.h"

#include <Eigen/Core>

#include <type_traits>

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
};

HOST_DEVICE inline bool
operator<(const IntersectionNormalUV &lhs,
          const IntersectionNormalUV &rhs) { /* do actual comparison */
  return lhs.intersection < rhs.intersection;
}
HOST_DEVICE inline bool operator>(const IntersectionNormalUV &lhs,
                      const IntersectionNormalUV &rhs) {
  return operator<(rhs, lhs);
}
HOST_DEVICE inline bool operator<=(const IntersectionNormalUV &lhs,
                       const IntersectionNormalUV &rhs) {
  return !operator>(lhs, rhs);
}
HOST_DEVICE inline bool operator>=(const IntersectionNormalUV &lhs,
                       const IntersectionNormalUV &rhs) {
  return !operator<(lhs, rhs);
}

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

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template <bool normals_and_uv>
HOST_DEVICE inline bool operator<(const BestIntersectionGeneral<normals_and_uv> &lhs,
                      const BestIntersectionGeneral<normals_and_uv>
                          &rhs) {
return lhs.intersection < rhs.intersection;
}
template <bool normals_and_uv>
HOST_DEVICE inline bool operator>(const BestIntersectionGeneral<normals_and_uv> &lhs,
                      const BestIntersectionGeneral<normals_and_uv> &rhs) {
  return operator<(rhs, lhs);
}
template <bool normals_and_uv>
HOST_DEVICE inline bool operator<=(const BestIntersectionGeneral<normals_and_uv> &lhs,
                       const BestIntersectionGeneral<normals_and_uv> &rhs) {
  return !operator>(lhs, rhs);
}
template <bool normals_and_uv>
HOST_DEVICE inline bool operator>=(const BestIntersectionGeneral<normals_and_uv> &lhs,
                       const BestIntersectionGeneral<normals_and_uv> &rhs) {
  return !operator<(lhs, rhs);
}

using BestIntersectionNormalUV = BestIntersectionGeneral<true>;
using BestIntersection = BestIntersectionGeneral<false>;
} // namespace detail
} // namespace ray
