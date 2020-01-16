#pragma once

#include "lib/span.h"
#include "ray/detail/accel/kdtree/kdtree.h"
#include "ray/detail/intersection/intersection.h"

#include <thrust/optional.h>

namespace ray {
namespace detail {
namespace accel {
namespace kdtree {
struct KDTreeRef {
  SpanSized<KDTreeNode<AABB>> nodes;
  unsigned num_shape;

  KDTreeRef(SpanSized<KDTreeNode<AABB>> nodes, unsigned num_shape)
      : nodes(nodes), num_shape(num_shape) {}

  KDTreeRef() {}

  template <typename SolveIndex>
  HOST_DEVICE void operator()(const Eigen::Vector3f &world_space_direction,
                              const Eigen::Vector3f &world_space_eye,
                              const thrust::optional<BestIntersection> &best,
                              const SolveIndex &solve_index) const {
    auto inv_direction = (1.0f / world_space_direction.array()).eval();

    struct StackData {
      unsigned node_index;
      uint8_t depth;

      HOST_DEVICE StackData(unsigned node_index, uint8_t depth)
          : node_index(node_index), depth(depth) {}

      HOST_DEVICE StackData() {}
    };

    if (nodes.size() != 0) {
      std::array<StackData, 64> node_stack;
      node_stack[0] = StackData(nodes.size() - 1, 0);
      uint8_t node_stack_size = 1;

      thrust::optional<std::array<unsigned, 2>> start_end = thrust::nullopt;
      unsigned current_shape_index = 0;

      while (node_stack_size != 0 || start_end.has_value()) {
        while (!start_end.has_value() && node_stack_size != 0) {
          const auto &stack_v = node_stack[node_stack_size - 1];

          const auto &current_node = nodes[stack_v.node_index];

          auto bounding_intersection =
              current_node.get_contents().solveBoundingIntersection(
                  world_space_eye, inv_direction);

          if (bounding_intersection.has_value() &&
              (!best.has_value() ||
               best->intersection > *bounding_intersection)) {
            current_node.case_split_or_data(
                [&](const KDTreeSplit &split) {
                  const uint8_t axis = stack_v.depth % 3;
                  const auto intersection_point =
                      world_space_eye[axis] +
                      world_space_direction[axis] * *bounding_intersection;
                  auto first = split.left_index;
                  auto second = split.right_index;

                  if (intersection_point > split.division_point) {
                    auto temp = first;
                    first = second;
                    second = temp;
                  }

                  uint8_t new_depth = stack_v.depth + 1;
                  node_stack[node_stack_size - 1] =
                      StackData(second, new_depth);
                  node_stack_size++;
                  node_stack[node_stack_size - 1] = StackData(first, new_depth);
                  node_stack_size++; // counter act --;
                },
                [&](const std::array<unsigned, 2> &data) {
                  start_end = thrust::make_optional(data);
                  current_shape_index = data[0];
                });
          }

          node_stack_size--;
        }

        if (start_end.has_value()) {
          auto local_end_shape = (*start_end)[1];
          unsigned shape_idx = current_shape_index;
          current_shape_index++;
          if (current_shape_index >= local_end_shape) {
            start_end = thrust::nullopt;
          }

          if (solve_index(shape_idx)) {
            return;
          }
        }
      }
    }
  }
};
} // namespace kdtree
} // namespace accel
} // namespace detail
} // namespace ray
