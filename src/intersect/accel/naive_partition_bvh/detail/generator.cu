#include "data_structure/copyable_to_vec.h"
#include "intersect/accel/naive_partition_bvh/detail/generator.h"
#include "lib/assert.h"
#include "lib/eigen_utils.h"

#include <dbg.h>

namespace intersect {
namespace accel {
namespace naive_partition_bvh {
template <ExecutionModel exec>
RefPerm<BVH<>>
NaivePartitionBVH<exec>::Generator::gen(const Settings &settings,
                                        SpanSized<Bounds> bounds) {
  settings_ = settings;
  settings_.num_objects_terminate =
      std::max(settings_.num_objects_terminate, 1u);

  NodeGroup<HostVector> nodes;
  HostVector<unsigned> idxs;

  auto make_ref_perm = [&](const auto &nodes_in) -> RefPerm<BVH<>> {
    // use nodes_ to ensure values are on cpu
    check_and_print_stats(nodes.get(tag_v<NodeItem::Value>),
                          nodes.get(tag_v<NodeItem::AABB>),
                          settings.bvh_settings, BVH<>::objects_vec_size);

    return {
        .ref =
            {
                .node_values = nodes_in.get(tag_v<NodeItem::Value>),
                .node_aabbs = nodes_in.get(tag_v<NodeItem::AABB>),
                .target_objects = settings.bvh_settings.target_objects,
            },
        .permutation = idxs,
    };
  };

  if (bounds.size() == 0) {
    return make_ref_perm(nodes_out_);
  }

  idxs.resize(bounds.size());

  for (unsigned i = 0; i < bounds.size(); ++i) {
    idxs[i] = i;
  }

  nodes.resize_all(1);
  auto node = create_node(bounds, idxs, nodes, 0, 0);
  nodes.set_all_tup(0, node);

  nodes.copy_to_other(nodes_out_);

  return make_ref_perm(nodes_out_);
}

template <ExecutionModel exec>
NodeTup NaivePartitionBVH<exec>::Generator::create_node(
    SpanSized<Bounds> bounds, SpanSized<unsigned> idxs,
    NodeGroup<HostVector> &nodes, unsigned start_idx, unsigned depth) {
  always_assert(bounds.size() == idxs.size());
  always_assert(!bounds.empty());

  if (bounds.size() <= settings_.num_objects_terminate) {
    auto total_bounds = get_bounding(bounds);
    return {{
        NodeValue(NodeValueRep{
            tag_v<NodeType::Items>,
            {
                .start = start_idx,
                .end = start_idx + static_cast<unsigned>(bounds.size()),
            }}),
        total_bounds,
    }};
  }

  const size_t k = bounds.size() / 2;
  always_assert(k != 0);

  const unsigned axis = depth % 3;

  kth_smallest(bounds, idxs, k, axis);

  unsigned new_depth = depth + 1;

  nodes.resize_all(nodes.size() + 2);
  unsigned left_idx = nodes.size() - 2;
  unsigned right_idx = nodes.size() - 1;

  auto left_node = create_node(bounds.slice_to(k), idxs.slice_to(k), nodes,
                               start_idx, new_depth);
  auto right_node = create_node(bounds.slice_from(k), idxs.slice_from(k), nodes,
                                start_idx + k, new_depth);

  nodes.set_all_tup(left_idx, left_node);
  nodes.set_all_tup(right_idx, right_node);

  return {{
      NodeValue(NodeValueRep{
          tag_v<NodeType::Split>,
          {.left_idx = left_idx, .right_idx = right_idx},
      }),
      nodes.get(tag_v<NodeItem::AABB>)[left_idx].union_other(
          nodes.get(tag_v<NodeItem::AABB>)[right_idx]),
  }};
}

void check_kth_smallest(SpanSized<Bounds> bounds, size_t k, uint8_t axis) {
  unsigned count_less = 0;
  unsigned count_less_eq = 0;
  float x = bounds[k].center[axis];
  for (const Bounds &bound : bounds) {
    if (bound.center[axis] < x) {
      ++count_less;
    }
    if (bound.center[axis] <= x) {
      ++count_less_eq;
    }
  }

  always_assert(count_less <= k && k <= count_less_eq);
}

// inspired by https://www.geeksforgeeks.org/quickselect-algorithm/
template <ExecutionModel exec>
void NaivePartitionBVH<exec>::Generator::kth_smallest(SpanSized<Bounds> bounds,
                                                      SpanSized<unsigned> idxs,
                                                      size_t k, uint8_t axis) {
  always_assert(bounds.size() == idxs.size());
  always_assert(!bounds.empty());

  // Partition the array around last
  // element and get position of pivot
  // element in sorted array
  size_t index = partition(bounds, idxs, axis);

  // If position next to k
  if (index == k || (index == 1 && k == 0) ||
      (index == bounds.size() - 2 && k == bounds.size() - 1)) {
#ifndef NDEBUG
    check_kth_smallest(bounds, k, axis);
#endif

    return;
  }

  // If position is more, recur
  // for left subarray
  if (index > k) {
    kth_smallest(bounds.slice_to(index), idxs.slice_to(index), k, axis);
  } else {
    // Else recur for right subarray
    kth_smallest(bounds.slice_from(index + 1), idxs.slice_from(index + 1),
                 k - index - 1, axis);
  }

#ifndef NDEBUG
  check_kth_smallest(bounds, k, axis);
#endif
}

// inspired by https://www.geeksforgeeks.org/quickselect-algorithm/
template <ExecutionModel exec>
unsigned NaivePartitionBVH<exec>::Generator::partition(SpanSized<Bounds> bounds,
                                                       SpanSized<unsigned> idxs,
                                                       uint8_t axis) {
  always_assert(bounds.size() == idxs.size());
  always_assert(!bounds.empty());

  if (bounds.size() == 1) {
    return 0;
  }

  unsigned end = bounds.size() - 1;
  float x = bounds[end].center[axis];
  size_t i = 0;
  for (size_t j = 0; j <= end - 1; j++) {
    if (bounds[j].center[axis] <= x) {
      std::swap(bounds[i], bounds[j]);
      std::swap(idxs[i], idxs[j]);
      i++;
    }
  }
  std::swap(bounds[i], bounds[end]);
  std::swap(idxs[i], idxs[end]);

  return i;
}

template <ExecutionModel exec>
AABB NaivePartitionBVH<exec>::Generator::get_bounding(
    SpanSized<Bounds> bounds) {
  AABB out = AABB::empty();

  for (const Bounds &bound : bounds) {
    out = out.union_other(bound.aabb);
  }

  return out;
}

template class NaivePartitionBVH<ExecutionModel::CPU>::Generator;
#ifndef CPU_ONLY
template class NaivePartitionBVH<ExecutionModel::GPU>::Generator;
#endif
} // namespace naive_partition_bvh
} // namespace accel
} // namespace intersect
