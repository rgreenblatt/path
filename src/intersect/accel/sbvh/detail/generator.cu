#include "data_structure/copyable_to_vec.h"
#include "intersect/accel/sbvh/detail/generator.h"
#include "intersect/triangle_impl.h"
#include "lib/assert.h"
#include "lib/eigen_utils.h"
#include "lib/info/timer.h"

#include <dbg.h>

namespace intersect {
namespace accel {
namespace sbvh {
template <ExecutionModel exec>
RefPerm<BVH<>>
SBVH<exec>::Generator::gen(const Settings &settings,
                           SpanSized<const Triangle> triangles_in) {
  Timer start;
  HostVector<Triangle> triangles(triangles_in.begin(), triangles_in.end());
  HostVector<unsigned> idxs(triangles_in.size());
  for (unsigned i = 0; i < triangles.size(); ++i) {
    idxs[i] = i;
  }

  NodeGroup<HostVector> nodes;
  nodes.resize_all(1);
  auto node =
      create_node(triangles, idxs, nodes,
                  settings.bvh_settings.traversal_per_intersect_cost, 0);
  nodes.set_all_tup(0, node);

  nodes.copy_to_other(nodes_);

  bvh::check_and_print_stats(nodes.get(tag_v<NodeItem::Value>),
                             nodes.get(tag_v<NodeItem::AABB>),
                             settings.bvh_settings, BVH<>::objects_vec_size);

  if (settings.bvh_settings.print_stats) {
    start.report("sbvh gen time");
  }

  return {
      .ref =
          {
              .node_values = nodes_.get(tag_v<NodeItem::Value>),
              .node_aabbs = nodes_.get(tag_v<NodeItem::AABB>),
              .target_objects = settings.bvh_settings.target_objects,
          },
      .permutation = idxs,
  };
}

template <ExecutionModel exec>
NodeTup SBVH<exec>::Generator::create_node(SpanSized<Triangle> triangles,
                                           SpanSized<unsigned> idxs,
                                           NodeGroup<HostVector> &nodes,
                                           float traversal_per_intersect_cost,
                                           unsigned start_idx) {
  always_assert(triangles.size() == idxs.size());

  AABB overall_aabb = AABB::empty();

  if (triangles.empty()) {
    return {{
        NodeValue(NodeValueRep{
            tag_v<NodeType::Items>,
            {.start = 0, .end = 0},
        }),
        overall_aabb,
    }};
  }

  ObjectSplitCandidate overall_best_split = {
      .base_cost = std::numeric_limits<float>::max(),
  };

  for (const auto &triangle : triangles) {
    overall_aabb = overall_aabb.union_other(triangle.bounds());
  }

  for (unsigned axis = 0; axis < 3; ++axis) {
    auto split = best_object_split(triangles, axis);
    if (split.base_cost < overall_best_split.base_cost) {
      overall_best_split = split;
    }
  }

  float surface_area = overall_aabb.surface_area();
  float best_cost = overall_best_split.base_cost / surface_area +
                    traversal_per_intersect_cost;

  if (best_cost >= triangles.size()) {
    return {{
        NodeValue(NodeValueRep{
            tag_v<NodeType::Items>,
            {
                .start = start_idx,
                .end = start_idx + static_cast<unsigned>(triangles.size()),
            },
        }),
        overall_aabb,
    }};
  }

  nodes.resize_all(nodes.size() + 2);
  unsigned left_idx = nodes.size() - 2;
  unsigned right_idx = nodes.size() - 1;

  HostVector<Triangle> old_triangles(triangles.begin(), triangles.end());
  HostVector<unsigned> old_idxs(idxs.begin(), idxs.end());

  for (unsigned i = 0; i < triangles.size(); ++i) {
    unsigned perm_idx = overall_best_split.perm[i];
    triangles[i] = old_triangles[perm_idx];
    idxs[i] = old_idxs[perm_idx];
  }

  unsigned split_point = overall_best_split.split_point;

  auto left_node =
      create_node(triangles.slice_to(split_point), idxs.slice_to(split_point),
                  nodes, traversal_per_intersect_cost, start_idx);
  auto right_node = create_node(
      triangles.slice_from(split_point), idxs.slice_from(split_point), nodes,
      traversal_per_intersect_cost, start_idx + split_point);

  nodes.set_all_tup(left_idx, left_node);
  nodes.set_all_tup(right_idx, right_node);

  return {{
      NodeValue(NodeValueRep{
          tag_v<NodeType::Split>,
          {
              .left_idx = left_idx,
              .right_idx = right_idx,
          },
      }),
      overall_aabb,
  }};
}

template <ExecutionModel exec>
ObjectSplitCandidate
SBVH<exec>::Generator::best_object_split(SpanSized<const Triangle> triangles_in,
                                         unsigned axis) {
  HostVector<Triangle> triangles(triangles_in.begin(), triangles_in.end());

  always_assert(!triangles.empty());

  auto sort_perm = sort_by_axis(triangles, axis);

  // inclusive
  HostVector<float> surface_areas_backward(triangles.size());
  AABB running_aabb_backward = AABB::empty();

  for (unsigned i = triangles.size() - 1;
       i != std::numeric_limits<unsigned>::max(); --i) {
    running_aabb_backward =
        running_aabb_backward.union_other(triangles[i].bounds());
    surface_areas_backward[i] = running_aabb_backward.surface_area();
  }

  float best_base_cost = std::numeric_limits<float>::max();
  unsigned best_split = 0;

  // exclusive
  AABB running_aabb = AABB::empty();

  for (unsigned i = 0; i < triangles.size(); ++i) {
    float surface_area_left = running_aabb.surface_area();
    float surface_area_right = surface_areas_backward[i];

    float base_cost =
        i * surface_area_left + (triangles.size() - i) * surface_area_right;
    if (base_cost < best_base_cost) {
      best_base_cost = base_cost;
      best_split = i;
    }

    running_aabb = running_aabb.union_other(triangles[i].bounds());
  }

  return {
      .perm = sort_perm,
      .split_point = best_split,
      .base_cost = best_base_cost,
  };
}

template <ExecutionModel exec>
HostVector<unsigned>
SBVH<exec>::Generator::sort_by_axis(SpanSized<Triangle> triangles,
                                    unsigned axis) {
  struct ValueIdx {
    float value;
    unsigned idx;
  };
  HostVector<ValueIdx> to_sort(triangles.size());

  for (unsigned i = 0; i < triangles.size(); ++i) {
    to_sort[i] = {
        .value = triangles[i].centroid()[axis],
        .idx = i,
    };
  }

  std::sort(to_sort.begin(), to_sort.end(),
            [](ValueIdx l, ValueIdx r) { return l.value < r.value; });

  HostVector<unsigned> out(triangles.size());

  HostVector<Triangle> old_triangles(triangles.begin(), triangles.end());

  for (unsigned i = 0; i < triangles.size(); ++i) {
    unsigned idx = to_sort[i].idx;
    out[i] = idx;
    triangles[i] = old_triangles[idx];
  }

  return out;
}

template class SBVH<ExecutionModel::CPU>::Generator;
#ifndef CPU_ONLY
template class SBVH<ExecutionModel::GPU>::Generator;
#endif
} // namespace sbvh
} // namespace accel
} // namespace intersect
