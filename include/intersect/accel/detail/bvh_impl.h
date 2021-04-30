#pragma once

#include "intersect/accel/add_idx.h"
#include "intersect/accel/detail/bvh.h"
#include "intersect/optional_min.h"
#include "lib/stack.h"
#include "lib/start_end.h"

namespace intersect {
namespace accel {
namespace detail {
template <IntersectableAtIdx F>
HOST_DEVICE inline AccelRet<F>
BVH::intersect_objects(const intersect::Ray &ray,
                       const F &intersectable_at_idx) const {
  AccelRet<F> best;

  if (nodes.size() == 0) {
    return std::nullopt;
  }

  // TODO: how important is this...
  Eigen::Vector3f direction_no_zeros = *ray.direction;
  auto remove_zero = [](float &v) {
    if (v == 0.0f || v == -0.0f) {
      v = 1e-20f;
    }
  };

  remove_zero(direction_no_zeros.x());
  remove_zero(direction_no_zeros.y());
  remove_zero(direction_no_zeros.z());

  auto inv_direction = (1.0f / direction_no_zeros.array()).eval();

  Stack<unsigned, 64> node_stack;
  node_stack.push(start_idx);

  StartEnd<unsigned> start_end = {.start = 0u, .end = 0u};

  while (!node_stack.empty()) {
    while (start_end.empty() && !node_stack.empty()) {
      const auto &current_node = nodes[node_stack.pop()];

      auto bounding_intersection =
          current_node.aabb.solve_bounding_intersection(ray.origin,
                                                        inv_direction);

      if (bounding_intersection.has_value() &&
          (!best.has_value() ||
           best->intersection_dist > *bounding_intersection)) {
        current_node.value.visit_tagged([&](auto tag, const auto &v) {
          if constexpr (tag == NodeType::Split) {
            node_stack.push(v.right_index);
            node_stack.push(v.left_index);
          } else {
            static_assert(tag == NodeType::Items);
            start_end = v;
          }
        });
      }
    }

    for (unsigned idx = start_end.start; idx < start_end.end; idx++) {
      auto intersection = intersectable_at_idx(idx, ray);
      best = optional_min(best, add_idx(intersection, idx));
    }

    start_end = {.start = 0u, .end = 0u};
  }

  return best;
}
} // namespace detail
} // namespace accel
} // namespace intersect
