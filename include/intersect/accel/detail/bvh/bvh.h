#pragma once

#include "intersect/accel/accel.h"
#include "intersect/accel/detail/bvh/node.h"
#include "intersect/accel/detail/bvh/settings.h"
#include "lib/span.h"
#include "lib/vector_group.h"

namespace intersect {
namespace accel {
namespace detail {
namespace bvh {
template <unsigned node_stack_size_in = 64, unsigned objects_vec_size_in = 10>
struct BVH {
  constexpr static unsigned node_stack_size = node_stack_size_in;

  // we should have target_objects + 1 <= objects_vec_size - (max_items - 1)
  constexpr static unsigned objects_vec_size = objects_vec_size_in;

  static_assert(node_stack_size > 0);
  static_assert(objects_vec_size > 0);

  SpanSized<const NodeValue> node_values;
  SpanSized<const AABB> node_aabbs;
  unsigned target_objects;

  template <IntersectableAtIdx F>
  HOST_DEVICE inline AccelRet<F>
  intersect_objects(const intersect::Ray &ray,
                    const F &intersectable_at_idx) const;
};

enum class NodeItem {
  Value,
  AABB,
};

template <template <typename> class VecT>
using NodeGroup = VectorGroup<VecT, NodeItem, NodeValue, AABB>;
using NodeTup = TaggedTuple<NodeItem, NodeValue, AABB>;

// These are independent functions to avoid template params
// (and because 'nodes' must be on the cpu!).

void check_and_print_stats(SpanSized<const NodeValue> node_values,
                           SpanSized<const AABB> node_aabbs, Settings settings,
                           unsigned objects_vec_size);

float sa_heurisitic_cost(SpanSized<const NodeValue> node_values,
                         SpanSized<const AABB> node_aabbs,
                         float traversal_per_intersect_cost);

static_assert(AccelRef<BVH<>>);
} // namespace bvh
} // namespace detail
} // namespace accel
} // namespace intersect
