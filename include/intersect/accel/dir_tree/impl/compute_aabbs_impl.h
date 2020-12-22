#pragma once

#include "lib/cuda/utils.h"
#include "lib/span.h"
#include "ray/detail/accel/dir_tree/bounding_points.h"
#include "ray/detail/accel/dir_tree/idx_aabb.h"

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
inline HOST_DEVICE void
compute_aabbs_impl(SpanSized<const Eigen::Projective3f> transforms,
                   unsigned transform_idx, Span<IdxAABB> aabbs,
                   SpanSized<const BoundingPoints> bounds, unsigned bound_idx) {
  if (transform_idx >= transforms.size() || bound_idx >= bounds.size()) {
    return;
  }

  Eigen::Vector3f min_bound = max_eigen_vec();
  Eigen::Vector3f max_bound = min_eigen_vec();

  for (const auto &point : bounds[bound_idx]) {
    auto transformed_point =
        apply_projective_point(point, transforms[transform_idx]);

    min_bound = min_bound.cwiseMin(transformed_point);
    max_bound = max_bound.cwiseMax(transformed_point);
  }

  aabbs[bound_idx + transform_idx * bounds.size()] =
      IdxAABB(bound_idx, AABB(min_bound, max_bound));
}
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
