#pragma once

#include "intersect/accel/add_idx.h"
#include "intersect/accel/detail/bvh/bvh.h"
#include "intersect/optional_min.h"
#include "lib/stack.h"

namespace intersect {
namespace accel {
namespace detail {
namespace bvh {
template <unsigned node_stack_size, unsigned objects_vec_size>
template <IntersectableAtIdx F>
HOST_DEVICE inline AccelRet<F>
BVH<node_stack_size, objects_vec_size>::intersect_objects(
    const intersect::Ray &ray, const F &intersectable_at_idx) const {
  debug_assert_assume(target_objects > 0);
  debug_assert_assume(target_objects <= objects_vec_size);

  AccelRet<F> best;

  if (nodes.empty()) {
    return std::nullopt;
  }

  auto inv_direction = get_inv_direction(*ray.direction);

  // reducing the size of these stack arrays appears to have no effect
  // on perf (so we can afford to be somewhat generous noting that
  // overflow would be UB)
  //
  // the size of objects_vec_size is checked by check_and_print_stats,
  // but node_stack_size isn't checked
  Stack<unsigned, node_stack_size> node_stack;
  node_stack.push(0);

  ArrayVec<unsigned, objects_vec_size> objects;

  while (!node_stack.empty()) {
    while (objects.size() < target_objects && !node_stack.empty()) {
      const auto &current_node = nodes[node_stack.pop()];

      auto bounding_intersection =
          current_node.aabb.solve_bounding_intersection(ray.origin,
                                                        inv_direction);

      if (bounding_intersection.has_value() &&
          (!best.has_value() ||
           best->intersection_dist > bounding_intersection->t_min)) {
        auto value = current_node.value.as_rep();

        value.visit_tagged([&](auto tag, const auto &v) {
          if constexpr (tag == NodeType::Split) {
            node_stack.push(v.right_idx);
            node_stack.push(v.left_idx);
          } else {
            static_assert(tag == NodeType::Items);
            // Maybe faster with if statement inside the loop? Probably doesn't
            // matter at all.
            auto se = v.start_end;
            if (v.is_for_extra) {
              for (unsigned idx = se.start; idx < se.end; ++idx) {
                objects.push_back(extra_idxs[idx]);
              }
            } else {
              for (unsigned idx = se.start; idx < se.end; ++idx) {
                objects.push_back(idx);
              }
            }
          }
        });
      }
    }

    if (objects.empty()) {
      debug_assert(node_stack.empty());
      break;
    }

    for (unsigned idx : objects) {
      auto intersection = intersectable_at_idx(idx, ray);
      best = optional_min(best, add_idx(intersection, idx));
    }

    objects.clear();
  }

  return best;
}
} // namespace bvh
} // namespace detail
} // namespace accel
} // namespace intersect
