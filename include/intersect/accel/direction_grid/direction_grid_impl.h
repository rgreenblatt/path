#pragma once

// TODO
#include "intersect/accel/add_idx.h"
#include "intersect/accel/direction_grid/direction_grid.h"

namespace intersect {
namespace accel {
namespace direction_grid {
namespace detail {

HOST_DEVICE inline unsigned DirectionGridRef::idx(
    const std::array<DirectionGridRef::IntersectionIdxs, 2> &intersections,
    unsigned grid) {
  std::array<unsigned, 2> idxs;
  for (unsigned i = 0; i < 2; ++i) {
    idxs[i] = grid * intersections[i].i + intersections[i].j;
  }

  bool first_is_larger = intersections[0].face > intersections[1].face;
  unsigned larger_face_idx = intersections[!first_is_larger].face;
  unsigned smaller_face_idx = intersections[first_is_larger].face;

  unsigned overall_face_idx =
      (larger_face_idx * (larger_face_idx - 1)) / 2 + smaller_face_idx;

  return grid * grid *
             (overall_face_idx * grid * grid + (idxs[!first_is_larger])) +
         (idxs[first_is_larger]);
}

template <IntersectableAtIdx F>
HOST_DEVICE inline AccelRet<F>
DirectionGridRef::intersect_objects(const intersect::Ray &ray,
                                    const F &intersectable_at_idx) const {
  auto inv_dir = get_inv_direction(*ray.direction);

  const auto &aabb = node_bounds[0];
  unsigned grid = node_grid[0];

  // NOTE this aabb approach is a modified copy paste from
  // solve_bounding_intersection.
  // This is probably not peak efficiency atm
  auto t_0 = (aabb.min_bound - ray.origin).cwiseProduct(inv_dir).eval();
  auto t_1 = (aabb.max_bound - ray.origin).cwiseProduct(inv_dir).eval();
  auto all_t_min = t_0.cwiseMin(t_1);
  auto all_t_max = t_0.cwiseMax(t_1);

  unsigned min_axis;
  unsigned max_axis;
  float overall_t_min = all_t_min.maxCoeff(&min_axis);
  float overall_t_max = all_t_max.minCoeff(&max_axis);

  if (overall_t_min > overall_t_max) {
    return std::nullopt;
  }

  bool max_bound_was_selected_for_t_min = t_1[min_axis] < t_0[min_axis];
  bool max_bound_was_selected_for_t_max = t_1[max_axis] >= t_0[max_axis];

  unsigned min_face_idx =
      unsigned(max_bound_was_selected_for_t_min) * 3 + min_axis;
  unsigned max_face_idx =
      unsigned(max_bound_was_selected_for_t_max) * 3 + max_axis;

  if (min_face_idx == max_face_idx) {
    debug_assert(max_bound_was_selected_for_t_min ==
                 max_bound_was_selected_for_t_max);
    debug_assert(min_axis == max_axis);
    // must be going through a corner or weird floating point nonsense
    return std::nullopt;
  }

  unsigned min_i_axis = (min_axis + 1) % 3;
  unsigned min_j_axis = (min_axis + 2) % 3;

  unsigned max_i_axis = (max_axis + 1) % 3;
  unsigned max_j_axis = (max_axis + 2) % 3;

  Eigen::Vector3f min_intersection_point =
      *ray.direction * overall_t_min + ray.origin;
  Eigen::Vector3f max_intersection_point =
      *ray.direction * overall_t_max + ray.origin;

  Eigen::Vector3f total = aabb.max_bound - aabb.min_bound;

  Eigen::Vector3<unsigned> min_idxs =
      (grid * (min_intersection_point - aabb.min_bound).array() / total.array())
          .floor()
          .template cast<unsigned>();
  Eigen::Vector3<unsigned> max_idxs =
      (grid * (max_intersection_point - aabb.min_bound).array() / total.array())
          .floor()
          .template cast<unsigned>();

  unsigned overall_idx = idx({IntersectionIdxs{
                                  .face = min_face_idx,
                                  .i = min_idxs[min_i_axis],
                                  .j = min_idxs[min_j_axis],
                              },
                              IntersectionIdxs{
                                  .face = max_face_idx,
                                  .i = max_idxs[max_i_axis],
                                  .j = max_idxs[max_j_axis],
                              }},
                             grid);
  auto start_end = direction_idxs[overall_idx];

  AccelRet<F> best;
  for (unsigned i = start_end.start; i < start_end.end; ++i) {
    unsigned obj_idx = overall_idxs[i];
    best = optional_min(best,
                        add_idx(intersectable_at_idx(obj_idx, ray), obj_idx));
  }

  return best;
}
} // namespace detail
} // namespace direction_grid
} // namespace accel
} // namespace intersect
