#pragma once

#include "ray/detail/accel/dir_tree/dir_tree_lookup_ref.h"
#include "ray/detail/projection.h"

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
template <typename SolveIndex>
inline HOST_DEVICE void
DirTreeLookupRef::operator()(const Eigen::Vector3f &world_space_direction,
                             const Eigen::Vector3f &world_space_eye,
                             const thrust::optional<BestIntersection> &best,
                             const SolveIndex &solve_index)

{
  const auto &[tree, flipped] = lookup_.getDirTree(world_space_direction);
  const auto transformed_dir =
      apply_projective_vec(world_space_direction, tree.transform);
  const auto transformed_eye =
      apply_projective_point(world_space_eye, tree.transform);
  /* tree.nodes */
  thrust::optional<std::array<unsigned, 2>> start_end;
  unsigned node_idx = 0;
  // approach:
  // - perform initial descent on eye point
  // - loop:
  //   - compute z at which we leave region
  //   - traverse/binary search to find z at which we enter region
  //   - keep intersecting for min < best intersection / min < leave region
  //   (appropriately transformed etc)
  //   - retraverse to new region
  while (!start_end.has_value()) {
  }
}
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
