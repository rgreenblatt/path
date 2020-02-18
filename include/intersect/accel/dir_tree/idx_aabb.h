#pragma once

#include "ray/detail/accel/aabb.h"

#include <iostream>

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
  friend std::ostream &operator<<(std::ostream &s, const IdxAABB &v) {
    s << "idx: " << v.idx << "\n"
      << "AABB: " << "\n" << v.aabb << "\n";

    return s;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
