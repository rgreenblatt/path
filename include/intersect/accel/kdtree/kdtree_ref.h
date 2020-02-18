#pragma once

#include "lib/span.h"
#include "ray/detail/accel/kdtree/kdtree.h"
#include "ray/detail/intersection/intersection.h"

#include <thrust/optional.h>

namespace ray {
namespace detail {
namespace accel {
namespace kdtree {
class KDTreeRef {
public:
  KDTreeRef(SpanSized<KDTreeNode<AABB>> nodes, unsigned /* num_shapes */)
      : nodes_(nodes) /* , num_shapes_(num_shapes) */
  {}

  KDTreeRef() {}

  template <typename SolveIndex>
  inline HOST_DEVICE void
  operator()(const Eigen::Vector3f &world_space_direction,
             const Eigen::Vector3f &world_space_eye,
             const thrust::optional<BestIntersection> &best,
             const SolveIndex &solve_index) const;

private:
  SpanSized<KDTreeNode<AABB>> nodes_;
  // unsigned num_shapes_; // could be useful for debugging, otherwise not
  // needed
};
} // namespace kdtree
} // namespace accel
} // namespace detail
} // namespace ray
