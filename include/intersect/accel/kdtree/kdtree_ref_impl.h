#include "ray/detail/accel/kdtree/kdtree_ref.h"

namespace ray {
namespace detail {
namespace accel {
namespace kdtree {
template <typename SolveIndex>
inline HOST_DEVICE void
KDTreeRef::operator()(const Eigen::Vector3f &world_space_direction,
                      const Eigen::Vector3f &world_space_eye,
                      const thrust::optional<BestIntersection> &best,
                      const SolveIndex &solve_index) const {
  Eigen::Vector3f direction_no_zeros = world_space_direction;
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

    HOST_DEVICE StackData(unsigned node_index, uint8_t depth)
        : node_index(node_index), depth(depth) {}

    HOST_DEVICE StackData() {}
  };

  if (nodes_.size() != 0) {
    std::array<StackData, 64> node_stack;
    node_stack[0] = StackData(nodes_.size() - 1, 0);
    uint8_t node_stack_size = 1;

    thrust::optional<std::array<unsigned, 2>> start_end = thrust::nullopt;
    unsigned current_shape_index = 0;

    while (node_stack_size != 0 || start_end.has_value()) {
      while (!start_end.has_value() && node_stack_size != 0) {
        const auto &stack_v = node_stack[node_stack_size - 1];

        const auto &current_node = nodes_[stack_v.node_index];

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
                node_stack[node_stack_size - 1] = StackData(second, new_depth);
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
} // namespace kdtree
} // namespace accel
} // namespace detail
} // namespace ray
