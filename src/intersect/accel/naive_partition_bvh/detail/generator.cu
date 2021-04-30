#include "intersect/accel/naive_partition_bvh/detail/generator.h"
#include "lib/assert.h"
#include "lib/eigen_utils.h"

#include <dbg.h>

namespace intersect {
namespace accel {
namespace naive_partition_bvh {
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

template <ExecutionModel exec>
bool NaivePartitionBVH<exec>::Generator::terminate_here(unsigned start,
                                                        unsigned end) {
  return end - start <= settings_.num_objects_terminate;
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

template <ExecutionModel exec>
Node NaivePartitionBVH<exec>::Generator::create_node(SpanSized<Bounds> bounds,
                                                     SpanSized<unsigned> idxs,
                                                     std::vector<Node> &nodes,
                                                     unsigned start_idx,
                                                     unsigned depth) {
  always_assert(bounds.size() == idxs.size());
  always_assert(!bounds.empty());

  if (bounds.size() <= settings_.num_objects_terminate) {
    auto total_bounds = get_bounding(bounds);
    return {
        .value = {tag_v<NodeType::Items>,
                  {
                      .start = start_idx,
                      .end = start_idx + static_cast<unsigned>(bounds.size()),
                  }},
        .aabb = total_bounds,
    };
  }

  const size_t k = bounds.size() / 2;
  always_assert(k != 0);

  const unsigned axis = depth % 3;

  kth_smallest(bounds, idxs, k, axis);

  unsigned new_depth = depth + 1;

  nodes.resize(nodes.size() + 2);
  unsigned left_idx = nodes.size() - 2;
  unsigned right_idx = nodes.size() - 1;

  nodes[left_idx] = create_node(bounds.slice_to(k), idxs.slice_to(k), nodes,
                                start_idx, new_depth);
  nodes[right_idx] = create_node(bounds.slice_from(k), idxs.slice_from(k),
                                 nodes, start_idx + k, new_depth);

  return {
      .value =
          {
              tag_v<NodeType::Split>,
              {.left_idx = left_idx, .right_idx = right_idx},
          },
      .aabb = nodes[left_idx].aabb.union_other(nodes[right_idx].aabb),
  };
}

template <ExecutionModel exec>
RefPerm<BVH> NaivePartitionBVH<exec>::Generator::gen(const Settings &settings,
                                                     SpanSized<Bounds> bounds) {
  settings_ = settings;
  settings_.num_objects_terminate =
      std::max(settings_.num_objects_terminate, 1u);

  nodes_out_.clear();
  indexes_.clear();

  if (bounds.size() == 0) {
    return {.ref = {.nodes = nodes_out_}, .permutation = indexes_};
  }

  indexes_.resize(bounds.size());

  for (unsigned i = 0; i < bounds.size(); ++i) {
    indexes_[i] = i;
  }

  nodes_.resize(1);
  nodes_[0] = create_node(bounds, indexes_, nodes_, 0, 0);

  if constexpr (exec == ExecutionModel::GPU) {
    nodes_out_.resize(nodes_.size());
    thrust::copy(nodes_.data(), nodes_.data() + nodes_.size(),
                 nodes_out_.begin());

    return {.ref = {.nodes = nodes_out_}, .permutation = indexes_};
  } else {
    return {.ref = {.nodes = nodes_}, .permutation = indexes_};
  }
}

template class NaivePartitionBVH<ExecutionModel::CPU>::Generator;
#ifndef CPU_ONLY
template class NaivePartitionBVH<ExecutionModel::GPU>::Generator;
#endif
} // namespace naive_partition_bvh
} // namespace accel
} // namespace intersect
