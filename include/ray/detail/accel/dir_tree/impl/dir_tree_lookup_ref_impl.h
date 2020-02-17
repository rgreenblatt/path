#pragma once

#include "lib/binary_search.h"
#include "lib/stack.h"
#include "ray/detail/accel/dir_tree/dir_tree_lookup_ref.h"
#include "ray/detail/accel/dir_tree/impl/sphere_partition_impl.h"
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
                             const SolveIndex &solve_index) const {
  // TODO: consider removing "flipped" output
  const auto &[tree_b, flipped] = lookup_.getDirTree(world_space_direction);
  const auto &tree = tree_b;
  const auto transformed_dir =
      apply_projective_vec(world_space_direction, tree.transform());
  const auto transformed_eye =
      apply_projective_point(world_space_eye, tree.transform());
  thrust::optional<DirTreeNode::StartEnd> start_end;

  struct IndexBound {
    float bound;
    unsigned index;
  };

  using StackT = Stack<IndexBound, 64>;
  StackT x_stack;
  StackT y_stack;

  bool x_positive = transformed_dir.x() >= 0;
  bool y_positive = transformed_dir.y() >= 0;
  bool z_positive = transformed_dir.z() >= 0;

  assert((flipped && !z_positive) || (!flipped && z_positive));

  float z_no_value = z_positive ? std::numeric_limits<float>::max()
                                : std::numeric_limits<float>::lowest();

  float z_change_respect_to_x = transformed_dir.z() / transformed_dir.x();
  float z_change_respect_to_y = transformed_dir.z() / transformed_dir.y();
  float x_change_respect_to_z = 1 / z_change_respect_to_x;
  float y_change_respect_to_z = 1 / z_change_respect_to_y;

  auto get_cmp = [](const bool is_positive) {
    return [=](const auto &v1, const auto &v2) {
      if (v1 < v2) {
        return is_positive;
      } else {
        return !is_positive;
      }
    };
  };

  unsigned min_max_idx = tree.min_max_idx();

  float x_end = (x_positive ? lookup_.x_max() : lookup_.x_min())[min_max_idx];
  float y_end = (y_positive ? lookup_.y_max() : lookup_.y_min())[min_max_idx];
  float z_end = (z_positive ? lookup_.z_max() : lookup_.z_min())[min_max_idx];

  float z_bound =
      std::min({(x_end - transformed_eye.x()) * z_change_respect_to_x,
                (y_end - transformed_eye.y()) * z_change_respect_to_y, z_end},
               get_cmp(z_positive));
  float x_bound = (z_bound - transformed_eye.z()) * x_change_respect_to_z +
                  transformed_eye.x();
  float y_bound = (z_bound - transformed_eye.z()) * y_change_respect_to_z +
                  transformed_eye.y();

  std::array<float, 2> traversal_values = {transformed_eye.x(),
                                           transformed_eye.y()};
  bool x_within_bound = false;
  bool y_within_bound = false;

  float start_z = transformed_eye.z();
  float final_z;

  auto intersect_to_z = [&](const float intersection_dist) {
    // TODO: I am pretty sure this is right, but check
    return intersection_dist * transformed_dir.z() + transformed_eye.z();
  };

  enum class Operation : uint8_t { X, Y, End };

  Operation next_operation = Operation::X;
  x_stack.push(IndexBound{NAN, tree.start_node_idx()});

  auto inclusive_search_values = z_positive
                                     ? lookup_.min_sorted_inclusive_maxes()
                                     : lookup_.max_sorted_inclusive_mins();
  auto start_values =
      z_positive ? lookup_.min_sorted_values() : lookup_.max_sorted_values();
  auto indexes =
      z_positive ? lookup_.min_sorted_indexes() : lookup_.max_sorted_indexes();

  // SPEED: look at how z_positive is used...

  while (next_operation != Operation::End) {
    bool is_x = next_operation == Operation::X;
    const auto &index_bound = is_x ? x_stack.pop() : y_stack.pop();
    unsigned node_idx = index_bound.index;

    x_within_bound = false;
    y_within_bound = false;

    while (!start_end.has_value()) {
      lookup_.nodes()[node_idx].visit([&](const auto &v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, DirTreeNode::Split>) {
          StackT &other_stack_ref = is_x ? y_stack : x_stack;
          bool &within_bound = is_x ? x_within_bound : y_within_bound;
          bool is_positive = is_x ? x_positive : y_positive;
          auto cmp = get_cmp(is_positive);
          float bound = is_x ? x_bound : y_bound;
          auto push = [&](unsigned idx) {
            other_stack_ref.push({v.split_point, idx});
          };
          auto is_within_bound_update = [&] {
            within_bound = within_bound || cmp(v.split_point, bound);

            return within_bound;
          };

          if (traversal_values[is_x ? 0 : 1] > v.split_point) {
            if (!is_positive && is_within_bound_update()) {
              push(v.left);
            }
            node_idx = v.right;
          } else {
            if (is_positive && is_within_bound_update()) {
              push(v.right);
            }
            node_idx = v.left;
          }

          is_x = !is_x;
        } else {
          static_assert(std::is_same_v<T, DirTreeNode::StartEnd>);
          start_end = v;

          // y stack corresponds to x split and x stack corresponds to y split
          float z_of_x_intersection =
              y_stack.size() == 0
                  ? z_no_value
                  : (y_stack.peek().bound - transformed_eye.x()) *
                            z_change_respect_to_x +
                        transformed_eye.z();
          float z_of_y_intersection =
              x_stack.size() == 0
                  ? z_no_value
                  : (x_stack.peek().bound - transformed_eye.y()) *
                            z_change_respect_to_y +
                        transformed_eye.z();

          using TupleT = std::tuple<float, Operation>;
          auto cmp = get_cmp(z_positive);
          const auto &[final_z_v, next_operation_v] =
              std::min<TupleT>({TupleT{z_of_x_intersection, Operation::Y},
                                TupleT{z_of_y_intersection, Operation::X},
                                TupleT{z_bound, Operation::End}},
                               [&](const TupleT &v1, const TupleT &v2) {
                                 return cmp(std::get<0>(v1), std::get<0>(v2));
                               });

          final_z = final_z_v;
          next_operation = next_operation_v;
        }
      });
    }

    // SPEED: guess!!!
    unsigned start_idx =
        binary_search(start_end->start, start_end->end, start_z,
                      inclusive_search_values, z_positive);

    for (unsigned i = start_idx; i < start_end->end; ++i) {
      if (start_values[i] > final_z) {
        if (z_positive) {
          break;
        }
      } else {
        if (!z_positive) {
          break;
        }
      }

      solve_index(indexes[i]);

      // SPEED: may be possible for this to be more efficient by
      // only updating when solution exists
      if (best.has_value()) {
        float z_of_intersection = intersect_to_z(best->intersection);
        if (z_of_intersection < final_z) {
          if (z_positive) {
            final_z = z_of_intersection;
            next_operation = Operation::End;
          }
        } else {
          if (!z_positive) {
            final_z = z_of_intersection;
            next_operation = Operation::End;
          }
        }
      }
    }

    float diff_z = final_z - transformed_eye.z();

    traversal_values[0] = diff_z * x_change_respect_to_z + transformed_eye.x();
    traversal_values[1] = diff_z * y_change_respect_to_z + transformed_eye.y();

    start_end = thrust::nullopt;

    start_z = final_z;
  }
}
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
