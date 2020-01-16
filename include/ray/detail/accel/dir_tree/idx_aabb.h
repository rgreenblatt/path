#pragma once

#include "ray/detail/accel/aabb.h"

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
struct IdxAABB {
  unsigned idx;
  AABB aabb;

  HOST_DEVICE IdxAABB(unsigned idx, const AABB &aabb) : idx(idx), aabb(aabb) {}

  HOST_DEVICE IdxAABB() {}

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
