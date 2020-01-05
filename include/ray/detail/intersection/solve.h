#pragma once

#include "lib/optional.h"
#include "ray/detail/intersection/intersection.h"
#include "ray/detail/intersection/shapes/solve.h"
#include "thrust/optional.h"

namespace ray {
namespace detail {
namespace intersection {
template <typename Accel, typename OnIntersection>
inline HOST_DEVICE void
solve(const Accel &accel, Span<const scene::ShapeData> shapes,
      const Eigen::Vector3f &world_space_eye,
      const Eigen::Vector3f &world_space_direction,
      const thrust::optional<BestIntersection> &best, const unsigned ignore_v,
      const OnIntersection &on_intersection) {
  auto solve_index = [&](unsigned shape_idx) {
    if (ignore_v == shape_idx) {
      return false;
    }

    return *optional_or(optional_map(shapes::get_intersection<false>(
                                         shapes, shape_idx, world_space_eye,
                                         world_space_direction),
                                     on_intersection),
                        thrust::optional<bool>{false});
  };

  accel(world_space_direction, world_space_eye, best, solve_index);
}
} // namespace intersection
} // namespace detail
} // namespace ray
