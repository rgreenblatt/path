#pragma once

#include "intersect/accel/add_idx.h"
#include "intersect/accel/kdtree/kdtree.h"
#include "lib/stack.h"

namespace intersect {
namespace accel {
namespace kdtree {
namespace detail {
template <Object O>
HOST_DEVICE inline AccelRet<O>
Ref::intersect_objects(const intersect::Ray &ray, Span<const O> objects) const {
  AccelRet<O> best;

  if (nodes_.size() == 0) {
    return nullopt_value;
  }

  // TODO: how important is this...
  Eigen::Vector3f direction_no_zeros = ray.direction;
  auto remove_zero = [](float &v) {
    if (v == 0.0f || v == -0.0f) {
      v = 1e-20f;
    }
  };

  remove_zero(direction_no_zeros.x());
  remove_zero(direction_no_zeros.y());
  remove_zero(direction_no_zeros.z());

  auto inv_direction = (1.0f / direction_no_zeros.array()).eval();

  struct StackData {
    unsigned node_index;
    uint8_t depth;
  };

  Stack<StackData, 64> node_stack;
  node_stack.push(StackData{unsigned(nodes_.size() - 1u), 0u});

  Optional<std::array<unsigned, 2>> start_end = nullopt_value;
  unsigned current_idx = 0;

  while (!node_stack.empty() || start_end.has_value()) {
    while (!start_end.has_value() && !node_stack.empty()) {
      const auto stack_v = node_stack.pop();

      const auto &current_node = nodes_[stack_v.node_index];

      auto bounding_intersection =
          current_node.get_contents().solveBoundingIntersection(ray.origin,
                                                                inv_direction);

      if (bounding_intersection.has_value() &&
          (!best.has_value() ||
           best->intersection_dist > *bounding_intersection)) {
        current_node.case_split_or_data(
            [&](const KDTreeSplit &split) {
              const uint8_t axis = stack_v.depth % 3;
              const auto intersection_point =
                  ray.origin[axis] +
                  ray.direction[axis] * *bounding_intersection;
              auto first = split.left_index;
              auto second = split.right_index;

              if (intersection_point > split.division_point) {
                auto temp = first;
                first = second;
                second = temp;
              }

              uint8_t new_depth = stack_v.depth + 1;
              node_stack.push(StackData{second, new_depth});
              node_stack.push(StackData{first, new_depth});
            },
            [&](const std::array<unsigned, 2> &data) {
              start_end = create_optional(data);
              current_idx = data[0];
            });
      }
    }

    if (start_end.has_value()) {
      for (unsigned idx = (*start_end)[0]; idx < (*start_end)[1]; idx++) {
        // TODO: SPEED
        // would it be better to enforce the same ordering everywhere somehow?
        unsigned global_idx = local_idx_to_global_idx_[idx];
        auto intersection = objects[global_idx].intersect(ray);
        best = optional_min(best, add_idx(intersection, global_idx));
      }
    }

    start_end = nullopt_value;
  }

  // convert to global index...
  return best;
}
} // namespace detail
} // namespace kdtree
} // namespace accel
} // namespace intersect
