#pragma once

#include "ray/detail/accel/dir_tree/dir_tree.h"
#include "ray/detail/intersection/intersection.h"
#include "lib/optional.h"

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
class DirTreeLookupRef {
public:
  DirTreeLookupRef(const DirTreeLookup &lookup) : lookup_(lookup) {}

  template <typename SolveIndex>
  inline HOST_DEVICE void
  operator()(const Eigen::Vector3f &world_space_direction,
             const Eigen::Vector3f &world_space_eye,
             const Optional<BestIntersection> &best,
             const SolveIndex &solve_index) const;

private:
  DirTreeLookup lookup_;
};
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
