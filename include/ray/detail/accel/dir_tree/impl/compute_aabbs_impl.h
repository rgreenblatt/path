#pragma once

#include "lib/cuda/utils.h"
#include "lib/span.h"
#include "ray/detail/accel/dir_tree/dir_tree.h"
#include "ray/detail/projection.h"

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
inline HOST_DEVICE void
compute_aabbs_impl(Span<Eigen::Projective3f> transforms, unsigned transform_idx,
                   unsigned num_transforms, Span<IdxAABB> aabbs,
                   Span<const BoundingPoints> bounds, unsigned bound_idx,
                   unsigned num_bounds) {
  if (transform_idx >= num_transforms || bound_idx >= num_bounds) {
    return;
  }

  Eigen::Vector3f min_bound = Eigen::Vector3f(
      std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
      std::numeric_limits<float>::max());
  Eigen::Vector3f max_bound =
      Eigen::Vector3f(std::numeric_limits<float>::lowest(),
                      std::numeric_limits<float>::lowest(),
                      std::numeric_limits<float>::lowest());

  for (const auto &point : bounds[bound_idx]) {
    auto transformed_point =
        apply_projective_point(point, transforms[transform_idx]);

    min_bound = min_bound.cwiseMin(transformed_point);
    max_bound = max_bound.cwiseMax(transformed_point);
  }

  aabbs[bound_idx + transform_idx * num_bounds] =
      IdxAABB(bound_idx, AABB(min_bound, max_bound));
}
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
