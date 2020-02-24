#pragma once

#include "intersect/accel/kdtree.h"
#include "lib/stack.h"

namespace intersect {

template <accel::KDTreeRef Ref> struct IntersectableImpl<Ref> {
  template <typename... T>
  static HOST_DEVICE inline auto intersect(const Ray &ray, const Ref &ref, T...vals) {
  using IntersectableO = IntersectableT<typename Ref::InstO>;
  using PrevInfoType = typename IntersectableO::template Intersection<T...>;
  using NewInfoType = AppendIndexInfoType<PrevInfoType>;
  using IntersectionOpT = IntersectionOp<NewInfoType>;

  IntersectionOpT best;

  if (ref.nodes_.size() == 0) {
    return IntersectionOpT{thrust::nullopt};
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
  node_stack.push(StackData{unsigned(ref.nodes_.size() - 1u), 0u});

  thrust::optional<std::array<unsigned, 2>> start_end = thrust::nullopt;
  unsigned current_idx = 0;

  while (!node_stack.empty() != 0 || start_end.has_value()) {
    while (!start_end.has_value() && node_stack.empty() != 0) {
      const auto stack_v = node_stack.pop();

      const auto &current_node = ref.nodes_[stack_v.node_index];

      auto bounding_intersection =
          current_node.get_contents().solveBoundingIntersection(ray.origin,
                                                                inv_direction);

      if (bounding_intersection.has_value() &&
          (!best.has_value() || best->intersection_dist > *bounding_intersection)) {
        current_node.case_split_or_data(
            [&](const accel::kdtree::KDTreeSplit &split) {
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
              start_end = thrust::make_optional(data);
              current_idx = data[0];
            });
      }
    }

    if (start_end.has_value()) {
      for (unsigned i = (*start_end)[0]; i < (*start_end)[1]; i++) {
        auto intersection =
            IntersectableO::intersect(ray, ref.objects_[i], vals...);
        best = optional_min(best, append_index(intersection, i));
      }
    }
  }

  return optional_map(best, [&](const auto& Intersection) -> intersect::Intersection<NewInfoType> {
      auto info = Intersection.info;
        info[info.size() - 1] = ref.local_idx_to_global_idx_[info[info.size() -1]] + ref.offset_;
        return {Intersection.intersection_dist, info};
      });
  }
};

template <accel::KDTreeRef Ref> struct BoundedImpl<Ref> {
  HOST_DEVICE static inline const accel::AABB &bounds(const Ref &ref) {
    return ref.aabb();
  }
};
} // namespace intersect
