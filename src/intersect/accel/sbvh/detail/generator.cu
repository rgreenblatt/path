#include "data_structure/copyable_to_vec.h"
#include "intersect/accel/sbvh/detail/generator.h"
#include "intersect/triangle_impl.h"
#include "lib/assert.h"
#include "lib/eigen_utils.h"

#include <dbg.h>

namespace intersect {
namespace accel {
namespace sbvh {
template <ExecutionModel exec>
RefPerm<BVH>
SBVH<exec>::Generator::gen(const Settings &settings,
                           SpanSized<const Triangle> triangles_in) {
  settings_ = settings;
  std::vector<Triangle> triangles(triangles_in.begin(), triangles_in.end());
  std::vector<unsigned> idxs(triangles_in.size());
  for (unsigned i = 0; i < triangles.size(); ++i) {
    idxs[i] = i;
  }

  std::vector<Node> nodes(1);
  nodes[0] = create_node(triangles, idxs, nodes, 0);

  dbg(sa_heurisitic_cost(nodes, 0));

  copy_to_vec(nodes, nodes_);

  return {
      .ref = {.nodes = nodes_, .start_idx = 0},
      .permutation = idxs,

  };
}

template <ExecutionModel exec>
Node SBVH<exec>::Generator::create_node(SpanSized<Triangle> triangles,
                                        SpanSized<unsigned> idxs,
                                        std::vector<Node> &nodes,
                                        unsigned start_idx) {
  always_assert(triangles.size() == idxs.size());

  AABB overall_aabb = AABB::empty();

  if (triangles.empty()) {
    return {
        .value = {tag_v<NodeType::Items>, {.start = 0, .end = 0}},
        .aabb = overall_aabb,
    };
  }

  ObjectSplitCandidate overall_best_split = {
      .cost = std::numeric_limits<float>::max(),
  };

  for (const auto &triangle : triangles) {
    overall_aabb = overall_aabb.union_other(triangle.bounds());
  }

  float surface_area = overall_aabb.surface_area();

  for (unsigned axis = 0; axis < 3; ++axis) {
    auto split = best_object_split(triangles, axis, surface_area);
    if (split.cost < overall_best_split.cost) {
      overall_best_split = split;
    }
  }

  if (overall_best_split.cost < triangles.size()) {
    return {
        .value =
            {
                tag_v<NodeType::Items>,
                {
                    .start = start_idx,
                    .end = start_idx + static_cast<unsigned>(triangles.size()),
                },
            },
        .aabb = overall_aabb,
    };
  } else {
    nodes.resize(nodes.size() + 2);
    unsigned left_idx = nodes.size() - 2;
    unsigned right_idx = nodes.size() - 1;

    std::vector<Triangle> old_triangles(triangles.begin(), triangles.end());
    std::vector<unsigned> old_idxs(idxs.begin(), idxs.end());

    for (unsigned i = 0; i < triangles.size(); ++i) {
      unsigned perm_idx = overall_best_split.perm[i];
      triangles[i] = old_triangles[perm_idx];
      idxs[i] = old_idxs[perm_idx];
    }

    unsigned split_point = overall_best_split.split_point;

    nodes[left_idx] = create_node(triangles.slice_to(split_point),
                                  idxs.slice_to(split_point), nodes, start_idx);
    nodes[right_idx] = create_node(triangles.slice_from(split_point),
                                   idxs.slice_from(split_point), nodes,
                                   start_idx + split_point);

    return {
        .value =
            {
                tag_v<NodeType::Split>,
                {
                    .left_index = left_idx,
                    .right_index = right_idx,
                },
            },
        .aabb = overall_aabb,
    };
  }
}

template <ExecutionModel exec>
ObjectSplitCandidate
SBVH<exec>::Generator::best_object_split(SpanSized<const Triangle> triangles_in,
                                         unsigned axis,
                                         float surface_area_above_node) {
  std::vector<Triangle> triangles(triangles_in.begin(), triangles_in.end());

  always_assert(triangles.size() > 0);

  auto sort_perm = sort_by_axis(triangles, axis);

  // inclusive
  std::vector<float> surface_areas_backward(triangles.size());
  AABB running_aabb_backward = AABB::empty();

  for (unsigned i = triangles.size() - 1;
       i != std::numeric_limits<unsigned>::max(); --i) {
    running_aabb_backward =
        running_aabb_backward.union_other(triangles[i].bounds());
    surface_areas_backward[i] = running_aabb_backward.surface_area();
  }

  float best_cost = std::numeric_limits<float>::max();
  unsigned best_split = 0;

  // exclusive
  AABB running_aabb = AABB::empty();

  for (unsigned i = 0; i < triangles.size(); ++i) {
    float surface_area_left = running_aabb.surface_area();
    float surface_area_right = surface_areas_backward[i];

    float cost = settings_.traversal_per_intersect_cost +
                 surface_area_left / surface_area_above_node +
                 surface_area_right / surface_area_above_node;
    if (cost < best_cost) {
      best_cost = cost;
      best_split = i;
    }

    running_aabb = running_aabb.union_other(triangles[i].bounds());
  }

  return {
      .perm = sort_perm,
      .split_point = best_split,
      .cost = best_cost,
  };
}

template <ExecutionModel exec>
std::vector<unsigned>
SBVH<exec>::Generator::sort_by_axis(SpanSized<Triangle> triangles,
                                    unsigned axis) {
  struct ValueIdx {
    float value;
    unsigned idx;
  };
  std::vector<ValueIdx> to_sort(triangles.size());

  for (unsigned i = 0; i < triangles.size(); ++i) {
    to_sort[i] = {
        .value = triangles[i].centroid()[axis],
        .idx = i,
    };
  }

  std::sort(to_sort.begin(), to_sort.end(),
            [](ValueIdx l, ValueIdx r) { return l.value < r.value; });

  std::vector<unsigned> out(triangles.size());

  std::vector<Triangle> old_triangles(triangles.begin(), triangles.end());

  for (unsigned i = 0; i < triangles.size(); ++i) {
    unsigned idx = to_sort[i].idx;
    out[i] = idx;
    triangles[i] = old_triangles[idx];
  }

  return out;
}

template <ExecutionModel exec>
float SBVH<exec>::Generator::sa_heurisitic_cost(SpanSized<const Node> nodes,
                                                unsigned start_node) {
  const auto &node = nodes[start_node];
  return node.value.visit_tagged([&](auto tag, const auto &value) {
    if constexpr (tag == NodeType::Items) {
      return value.size();
    } else {
      static_assert(tag == NodeType::Split);
      auto get_cost = [&](unsigned idx) {
        return sa_heurisitic_cost(nodes, idx) * nodes[idx].aabb.surface_area() /
               node.aabb.surface_area();
      };

      return settings_.traversal_per_intersect_cost +
             get_cost(value.left_index) + get_cost(value.right_index);
    }
  });
}

template class SBVH<ExecutionModel::CPU>::Generator;
#ifndef CPU_ONLY
template class SBVH<ExecutionModel::GPU>::Generator;
#endif
} // namespace sbvh
} // namespace accel
} // namespace intersect
