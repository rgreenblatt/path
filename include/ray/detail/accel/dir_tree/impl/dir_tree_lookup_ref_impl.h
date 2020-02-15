#pragma once

#include "lib/stack.h"
#include "lib/binary_search.h"
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

  bool x_positive = transformed_dir.x() > 0;
  bool y_positive = transformed_dir.y() > 0;
  bool z_positive = transformed_dir.z() > 0;

  assert((flipped && !z_positive) || (!flipped && z_positive));

  float z_no_value = z_positive ? std::numeric_limits<float>::max()
                                : std::numeric_limits<float>::lowest();

  float z_change_respect_to_x = transformed_dir.z() / transformed_dir.x();
  float z_change_respect_to_y = transformed_dir.z() / transformed_dir.y();

  unsigned min_max_idx = tree.min_max_idx();
  float x_bound = (x_positive ? lookup_.x_max() : lookup_.x_min())[min_max_idx];
  float y_bound = (y_positive ? lookup_.y_max() : lookup_.y_min())[min_max_idx];
  float z_bound = (z_positive ? lookup_.z_max() : lookup_.z_min())[min_max_idx];

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

  enum class Operation { X, Y, End };

  Operation next_operation = Operation::X;
  x_stack.push(IndexBound{z_bound, tree.start_node_idx()});

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
    auto &index_bound = is_x ? x_stack.pop() : y_stack.pop();
    unsigned node_idx = index_bound.index;

    while (!start_end.has_value()) {
      lookup_.nodes()[node_idx].visit([&](const auto &v) {
        if constexpr (std::is_same_v<v, DirTreeNode::Split>) {
          StackT &stack_ref = is_x ? x_stack : y_stack;
          bool &within_bound = is_x ? x_within_bound : y_within_bound;
          bool is_positive = is_x ? x_positive : y_positive;
          float bound = is_x ? x_bound : y_bound;
          auto push = [&] { stack_ref.push({v.split_point, node_idx}); };
          auto is_within_bound_update = [&] {
            within_bound =
                within_bound ||
                (is_positive ? v.split_point < bound : v.split_point > bound);

            return within_bound;
          };

          if (traversal_values[is_x] > v.split_point) {
            if (!is_positive && is_within_bound_update()) {
              push();
            }
            node_idx = v.right;
          } else {
            if (is_positive && is_within_bound_update()) {
              push();
            }
            node_idx = v.left;
          }

          is_x = !is_x;
        } else {
          start_end = v;
          float z_of_x_intersection =
              x_stack.size() == 0
                  ? z_no_value
                  : (x_stack.peek().bound - transformed_eye.x()) *
                        z_change_respect_to_x;
          float z_of_y_intersection =
              y_stack.size() == 0
                  ? z_no_value
                  : (y_stack.peek().bound - transformed_eye.y()) *
                        z_change_respect_to_y;

          using TupleT = std::tuple<float, uint8_t>;
          const auto &[final_z_v, next_operation_v] =
              std::min({{z_of_x_intersection, Operation::X},
                        {z_of_y_intersection, Operation::Y},
                        {z_bound, Operation::End}},
                       [&](const TupleT &v1, const TupleT &v2) {
                         if (std::get<0>(v1) < std::get<0>(v2)) {
                           return z_positive;
                         } else {
                           return !z_positive;
                         }
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

    start_z = final_z;
  }
}
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
