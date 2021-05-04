#include "data_structure/copyable_to_vec.h"
#include "intersect/accel/detail/clipped_triangle_impl.h"
#include "intersect/accel/sbvh/detail/generator.h"
#include "intersect/triangle_impl.h"
#include "lib/assert.h"
#include "lib/eigen_utils.h"
#include "lib/info/timer.h"

#include <boost/function_output_iterator.hpp>

#include <unordered_set>

namespace intersect {
namespace accel {
namespace sbvh {
template <ExecutionModel exec>
RefPerm<BVH<>>
SBVH<exec>::Generator::gen(const Settings &settings,
                           SpanSized<const Triangle> triangles_in) {
  Timer start;
  HostVector<ClippedTriangle> triangles(triangles_in.size());
  HostVector<unsigned> idxs(triangles.size());
  for (unsigned i = 0; i < triangles.size(); ++i) {
    triangles[i] = ClippedTriangle(triangles_in[i]);
    idxs[i] = i;
  }
  HostVector<std::optional<unsigned>> final_idxs_for_dups(triangles.size(),
                                                          std::nullopt);

  HostVector<Node> nodes(1);
  HostVector<unsigned> extra_idxs;
  nodes[0] =
      create_node(triangles, idxs, final_idxs_for_dups, nodes, extra_idxs,
                  settings.bvh_settings.traversal_per_intersect_cost,
                  settings.use_spatial_splits, 0);

  copy_to_vec(nodes, nodes_);
  copy_to_vec(extra_idxs, extra_idxs_);

  bvh::check_and_print_stats(nodes, settings.bvh_settings,
                             BVH<>::objects_vec_size);

  if (settings.bvh_settings.print_stats) {
    start.report("sbvh gen time");
  }

#ifndef NDEBUG
  std::unordered_set<unsigned> idxs_check(idxs.begin(), idxs.end());
  debug_assert(idxs_check.size() == idxs.size());
  if (!idxs.empty()) {
    debug_assert(*std::min_element(idxs.begin(), idxs.end()) == 0);
    debug_assert(*std::max_element(idxs.begin(), idxs.end()) ==
                 idxs.size() - 1);
  }
#endif

  return {
      .ref =
          {
              .nodes = nodes_,
              .extra_idxs = extra_idxs_,
              .target_objects = settings.bvh_settings.target_objects,
          },
      .permutation = idxs,
  };
}

template <ExecutionModel exec>
Node SBVH<exec>::Generator::create_node(
    SpanSized<ClippedTriangle> triangles, SpanSized<unsigned> idxs,
    SpanSized<std::optional<unsigned>> final_idxs_for_dups,
    HostVector<Node> &nodes, HostVector<unsigned> &extra_idxs,
    float traversal_per_intersect_cost, bool use_spatial_splits,
    unsigned start_idx) {
  always_assert(triangles.size() == idxs.size());
  always_assert(triangles.size() == final_idxs_for_dups.size());
  always_assert(!triangles.empty());

  SplitCandidate overall_best_split = {
      .base_cost = std::numeric_limits<float>::max(),
  };

  for (unsigned axis = 0; axis < 3; ++axis) {
    auto best_split_for_axis = best_object_split(triangles.as_const(), axis);
    if (use_spatial_splits) {
      best_split_for_axis = std::min(
          best_split_for_axis, best_spatial_split(triangles.as_const(), axis));
    }
    overall_best_split = std::min(overall_best_split, best_split_for_axis);
  }

  AABB overall_aabb = AABB::empty();

  for (const auto &triangle : triangles) {
    overall_aabb = overall_aabb.union_other(triangle.bounds);
  }

  float surface_area = overall_aabb.surface_area();
  float best_cost = overall_best_split.base_cost / surface_area +
                    traversal_per_intersect_cost;

  if (best_cost >= triangles.size()) {
    HostVector<unsigned> actual_final_idxs(final_idxs_for_dups.size());
    unsigned running_idx = start_idx;
    for (unsigned i = 0; i < final_idxs_for_dups.size(); ++i) {
      if (final_idxs_for_dups[i].has_value()) {
        actual_final_idxs[i] = *final_idxs_for_dups[i];
      } else {
        actual_final_idxs[i] = running_idx;
        ++running_idx;
      }
    }
    bool use_start_end = true;
    for (unsigned i = 0; i < actual_final_idxs.size() - 1; ++i) {
      use_start_end = use_start_end &&
                      actual_final_idxs[i + 1] > actual_final_idxs[i] &&
                      actual_final_idxs[i + 1] - actual_final_idxs[i] == 1;
    }

    auto item = [&]() -> Items {
      if (!use_start_end) {
        for (unsigned actual_idx : actual_final_idxs) {
          extra_idxs.push_back(actual_idx);
        }

        return {
            .is_for_extra = true,
            .start_end =
                {
                    .start =
                        unsigned(extra_idxs.size() - actual_final_idxs.size()),
                    .end = unsigned(extra_idxs.size()),
                },
        };
      } else {
        debug_assert(actual_final_idxs[0] + unsigned(triangles.size()) ==
                     actual_final_idxs[actual_final_idxs.size() - 1] + 1);
        return {
            .is_for_extra = false,
            .start_end =
                {
                    .start = actual_final_idxs[0],
                    .end = actual_final_idxs[0] + unsigned(triangles.size()),
                },
        };
      }
    }();

    return {
        .value = NodeValue(NodeValueRep{tag_v<NodeType::Items>, item}),
        .aabb = overall_aabb,
    };
  }

  nodes.resize(nodes.size() + 2);
  unsigned left_idx = nodes.size() - 2;
  unsigned right_idx = nodes.size() - 1;

  HostVector<ClippedTriangle> old_triangles(triangles.begin(), triangles.end());
  HostVector<unsigned> old_idxs(idxs.begin(), idxs.end());
  HostVector<std::optional<unsigned>> old_final_idxs_for_dups(
      final_idxs_for_dups.begin(), final_idxs_for_dups.end());

  overall_best_split.item.visit_tagged([&](auto tag, const auto &split) {
    if constexpr (tag == SplitType::Object) {
      for (unsigned i = 0; i < triangles.size(); ++i) {
        unsigned perm_idx = split.perm[i];
        triangles[i] = old_triangles[perm_idx];
        idxs[i] = old_idxs[perm_idx];
        final_idxs_for_dups[i] = old_final_idxs_for_dups[perm_idx];
      }

      unsigned split_point = split.split_point;

      unsigned num_in_order_before_split = 0;
      for (unsigned i = 0; i < split_point; ++i) {
        if (!final_idxs_for_dups[i].has_value()) {
          ++num_in_order_before_split;
        }
      }

      nodes[left_idx] = create_node(
          triangles.slice_to(split_point), idxs.slice_to(split_point),
          final_idxs_for_dups.slice_to(split_point), nodes, extra_idxs,
          traversal_per_intersect_cost, use_spatial_splits, start_idx);
      nodes[right_idx] = create_node(
          triangles.slice_from(split_point), idxs.slice_from(split_point),
          final_idxs_for_dups.slice_from(split_point), nodes, extra_idxs,
          traversal_per_intersect_cost, use_spatial_splits,
          start_idx + num_in_order_before_split);
    } else {
      always_assert(split.left_triangles.size() == split.left_bounds.size());
      always_assert(split.right_triangles.size() == split.right_bounds.size());

      for (unsigned i = 0; i < split.left_triangles.size(); ++i) {
        unsigned perm_idx = split.left_triangles[i];
        triangles[i] = old_triangles[perm_idx];
        triangles[i].bounds = split.left_bounds[i];
        idxs[i] = old_idxs[perm_idx];
        final_idxs_for_dups[i] = old_final_idxs_for_dups[perm_idx];
      }

      unsigned split_point = split.left_triangles.size();

      nodes[left_idx] = create_node(
          triangles.slice_to(split_point), idxs.slice_to(split_point),
          final_idxs_for_dups.slice_to(split_point), nodes, extra_idxs,
          traversal_per_intersect_cost, use_spatial_splits, start_idx);

      std::unordered_set<unsigned> dups;
      std::set_intersection(
          split.left_triangles.begin(), split.left_triangles.end(),
          split.right_triangles.begin(), split.right_triangles.end(),
          std::inserter(dups, dups.begin()));

      std::unordered_map<unsigned, unsigned> dup_idx_to_right_idx;

      for (unsigned i = 0; i < split.right_triangles.size(); ++i) {
        unsigned actual_idx = split.right_triangles[i];
        if (dups.contains(actual_idx)) {
          dup_idx_to_right_idx.insert({old_idxs[actual_idx], i});
        }
      }

      HostVector<ClippedTriangle> triangles_for_right(
          split.right_triangles.size());

      // just "enumerate"
      HostVector<unsigned> idxs_for_right(split.right_triangles.size());

      HostVector<std::optional<unsigned>> final_idxs_for_dups_for_right(
          split.right_triangles.size());

      for (unsigned i = 0; i < split.right_triangles.size(); ++i) {
        unsigned perm_idx = split.right_triangles[i];
        triangles_for_right[i] = old_triangles[perm_idx];
        triangles_for_right[i].bounds = split.right_bounds[i];
        idxs_for_right[i] = i;
        final_idxs_for_dups_for_right[i] = old_final_idxs_for_dups[perm_idx];
      }

      unsigned num_in_order_before_split = 0;
      for (unsigned i = 0; i < split.left_triangles.size(); ++i) {
        // this is reordered by the previous call to create_node
        // same with final_idxs_for_dups (below)
        unsigned idx = idxs[i];

        auto it = dup_idx_to_right_idx.find(idx);
        if (it != dup_idx_to_right_idx.end()) {
          unsigned right_idx = it->second;

          debug_assert(final_idxs_for_dups_for_right[right_idx] ==
                       final_idxs_for_dups[i]);

          if (!final_idxs_for_dups_for_right[right_idx].has_value()) {
            final_idxs_for_dups_for_right[right_idx] =
                start_idx + num_in_order_before_split;
          }
        }
        if (!final_idxs_for_dups[i].has_value()) {
          ++num_in_order_before_split;
        }
      }

      nodes[right_idx] = create_node(
          triangles_for_right, idxs_for_right, final_idxs_for_dups_for_right,
          nodes, extra_idxs, traversal_per_intersect_cost, use_spatial_splits,
          start_idx + num_in_order_before_split);

      unsigned overall_idx = split.left_triangles.size();
      for (unsigned i = 0; i < split.right_triangles.size(); ++i) {
        unsigned prev_step_idx = idxs_for_right[i];
        if (dups.contains(split.right_triangles[prev_step_idx])) {
          continue;
        }

        idxs[overall_idx] = old_idxs[split.right_triangles[prev_step_idx]];
        final_idxs_for_dups[overall_idx] =
            old_final_idxs_for_dups[split.right_triangles[prev_step_idx]];

        ++overall_idx;
      }
    }
  });

  return {
      .value = NodeValue(NodeValueRep{
          tag_v<NodeType::Split>,
          {
              .left_idx = left_idx,
              .right_idx = right_idx,
          },
      }),
      .aabb = overall_aabb,
  };
}

template <ExecutionModel exec>
SplitCandidate SBVH<exec>::Generator::best_object_split(
    SpanSized<const ClippedTriangle> triangles_in, unsigned axis) {
  HostVector<ClippedTriangle> triangles(triangles_in.begin(),
                                        triangles_in.end());

  always_assert(!triangles.empty());

  auto sort_perm = sort_by_axis(triangles, axis);

  // inclusive
  HostVector<float> surface_areas_backward(triangles.size());
  AABB running_aabb_backward = AABB::empty();

  for (unsigned i = triangles.size() - 1;
       i != std::numeric_limits<unsigned>::max(); --i) {
    running_aabb_backward =
        running_aabb_backward.union_other(triangles[i].bounds);
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

    running_aabb = running_aabb.union_other(triangles[i].bounds);
  }

  return {
      .base_cost = best_base_cost,
      .item =
          {
              tag_v<SplitType::Object>,
              {
                  .perm = sort_perm,
                  .split_point = best_split,
              },
          },
  };
}

template <ExecutionModel exec>
HostVector<unsigned>
SBVH<exec>::Generator::sort_by_axis(SpanSized<ClippedTriangle> triangles,
                                    unsigned axis) {
  struct ValueIdx {
    float value;
    unsigned idx;
  };
  HostVector<ValueIdx> to_sort(triangles.size());

  for (unsigned i = 0; i < triangles.size(); ++i) {
    to_sort[i] = {
        // TODO: somehow use the triangle?
        .value = triangles[i].bounds.centroid()[axis],
        .idx = i,
    };
  }

  std::sort(to_sort.begin(), to_sort.end(),
            [](ValueIdx l, ValueIdx r) { return l.value < r.value; });

  HostVector<unsigned> out(triangles.size());

  HostVector<ClippedTriangle> old_triangles(triangles.begin(), triangles.end());

  for (unsigned i = 0; i < triangles.size(); ++i) {
    unsigned idx = to_sort[i].idx;
    out[i] = idx;
    triangles[i] = old_triangles[idx];
  }

  return out;
}

template <ExecutionModel exec>
SplitCandidate SBVH<exec>::Generator::best_spatial_split(
    SpanSized<const ClippedTriangle> triangles, unsigned axis) {
  always_assert(!triangles.empty());

  AABB overall_aabb = AABB::empty();
  for (const ClippedTriangle &triangle : triangles) {
    overall_aabb = overall_aabb.union_other(triangle.bounds);
  }

  // TODO: make param?
  unsigned num_divisions = triangles.size() * 2;
  // doesn't include the furthest left and right edges
  unsigned num_inside_edges = num_divisions - 1;

  float total = overall_aabb.max_bound[axis] - overall_aabb.min_bound[axis];

  struct BinInfo {
    AABB aabb = AABB::empty();
    unsigned entries = 0;
    unsigned exits = 0;
  };

  HostVector<BinInfo> bin_infos(num_divisions);
  HostVector<HostVector<unsigned>> triangles_crossing_edge(num_inside_edges);

  auto get_split_point = [&](unsigned loc) {
    float frac = float(loc) / num_divisions;
    if (loc == num_divisions) {
      // reduce float error (actually needed...)
      return overall_aabb.max_bound[axis];
    }
    return frac * total + overall_aabb.min_bound[axis];
  };

  for (unsigned triangle_idx = 0; triangle_idx < triangles.size();
       ++triangle_idx) {
    const ClippedTriangle &triangle = triangles[triangle_idx];

    auto bounds = triangle.bounds;
    float min_axis = bounds.min_bound[axis];
    float max_axis = bounds.max_bound[axis];

    float min_prop = (min_axis - overall_aabb.min_bound[axis]) / total;
    float max_prop = (max_axis - overall_aabb.min_bound[axis]) / total;

    // min on both to handle case where triangle is on max edge
    unsigned min_loc = std::min(unsigned(std::floor(min_prop * num_divisions)),
                                num_divisions - 1);
    unsigned max_loc = std::min(unsigned(std::floor(max_prop * num_divisions)),
                                num_divisions - 1);

    // special case some floating point issues....
    // maybe find a better solution later...
    if (get_split_point(min_loc) > bounds.min_bound[axis]) {
      debug_assert_assume(min_loc > 0);
      --min_loc;
    }
    if (get_split_point(max_loc + 1) < bounds.max_bound[axis]) {
      debug_assert_assume(max_loc < num_divisions - 1);
      ++max_loc;
    }
    if (min_loc != max_loc) {
      if (get_split_point(max_loc) >= max_axis) {
        --max_loc;
      }
      if (get_split_point(min_loc + 1) <= min_axis) {
        ++min_loc;
      }
    }

    ++bin_infos[min_loc].entries;
    ++bin_infos[max_loc].exits;

    for (unsigned loc = min_loc; loc <= max_loc; ++loc) {
      if (loc != max_loc) {
        unsigned edge_after = loc;
        triangles_crossing_edge[edge_after].push_back(triangle_idx);
      }

      AABB &loc_aabb = bin_infos[loc].aabb;

      float split_left = get_split_point(loc);
      float split_right = get_split_point(loc + 1);
      loc_aabb = loc_aabb.union_other(
          triangle.new_bounds(split_left, split_right, axis));
    }
  }

#ifndef NDEBUG
  {
    unsigned total_entries = 0;
    unsigned total_exits = 0;
    for (const BinInfo &bin_info : bin_infos) {
      total_entries += bin_info.entries;
      total_exits += bin_info.exits;
    }
    debug_assert(total_entries == triangles.size());
    debug_assert(total_exits == triangles.size());
  }
#endif

  // inclusive
  HostVector<AABB> aabbs_backward(bin_infos.size());
  AABB running_aabb_backward = AABB::empty();

  for (unsigned i = bin_infos.size() - 1;
       i != std::numeric_limits<unsigned>::max(); --i) {
    running_aabb_backward =
        running_aabb_backward.union_other(bin_infos[i].aabb);
    aabbs_backward[i] = running_aabb_backward;
  }

  float best_base_cost = std::numeric_limits<float>::max();
  unsigned best_edge = 0;
  std::unordered_set<unsigned> best_left_only_tris;
  std::unordered_set<unsigned> best_right_only_tris;

  unsigned total_left_overall = 0;
  unsigned total_right_overall = triangles.size();
  AABB running_aabb = AABB::empty();
  for (unsigned edge = 0; edge < num_divisions - 1; ++edge) {
    unsigned loc_left = edge;
    unsigned loc_right = edge + 1;

    total_left_overall += bin_infos[loc_left].entries;
    total_right_overall -= bin_infos[loc_left].exits;

    unsigned total_left = total_left_overall;
    unsigned total_right = total_right_overall;

    running_aabb = running_aabb.union_other(bin_infos[loc_left].aabb);

    AABB left_aabb = running_aabb;
    AABB right_aabb = aabbs_backward[loc_right];

    float surface_area_left = left_aabb.surface_area();
    float surface_area_right = right_aabb.surface_area();

    float base_cost =
        total_left * surface_area_left + total_right * surface_area_right;

    std::unordered_set<unsigned> left_only_tris;
    std::unordered_set<unsigned> right_only_tris;

    SpanSized<const unsigned> triangle_idxs = triangles_crossing_edge[edge];
    for (unsigned triangle_idx : triangle_idxs) {
      const ClippedTriangle &triangle = triangles[triangle_idx];

      const AABB left_only = left_aabb.union_other(triangle.bounds);
      const AABB right_only = right_aabb.union_other(triangle.bounds);

      float base_cost_left_only = total_left * left_only.surface_area() +
                                  (total_right - 1) * surface_area_right;
      float base_cost_right_only = (total_left - 1) * surface_area_left +
                                   total_right * right_only.surface_area();

      if (std::min(base_cost_left_only, base_cost_right_only) < base_cost) {
        if (base_cost_left_only < base_cost_right_only) {
          total_right -= 1;
          base_cost = base_cost_left_only;
          surface_area_left = left_only.surface_area();
          left_aabb = left_only;
          left_only_tris.insert(triangle_idx);
        } else {
          total_left -= 1;
          base_cost = base_cost_right_only;
          surface_area_right = right_only.surface_area();
          right_aabb = right_only;
          right_only_tris.insert(triangle_idx);
        }
      }
    }

    if (base_cost < best_base_cost) {
      best_base_cost = base_cost;
      best_edge = edge;
      best_left_only_tris = left_only_tris;
      best_right_only_tris = right_only_tris;
    }
  }

  debug_assert(best_base_cost != std::numeric_limits<float>::max());

  // we need to to rebuild the AABBs because the unsplitting may
  // result in overly large AABBs
  AABB left_aabb = AABB::empty();
  AABB right_aabb = AABB::empty();
  HostVector<unsigned> left_triangles;
  HostVector<unsigned> right_triangles;
  HostVector<AABB> left_bounds;
  HostVector<AABB> right_bounds;

  for (unsigned triangle_idx = 0; triangle_idx < triangles.size();
       ++triangle_idx) {
    const ClippedTriangle &triangle = triangles[triangle_idx];

    if (best_left_only_tris.contains(triangle_idx)) {
      left_aabb = left_aabb.union_other(triangle.bounds);
      left_bounds.push_back(triangle.bounds);
      left_triangles.push_back(triangle_idx);
      continue;
    }
    if (best_right_only_tris.contains(triangle_idx)) {
      right_aabb = right_aabb.union_other(triangle.bounds);
      right_bounds.push_back(triangle.bounds);
      right_triangles.push_back(triangle_idx);
      continue;
    }

    float split_point = get_split_point(best_edge + 1);

    bool to_left = triangle.bounds.min_bound[axis] < split_point;
    bool to_right = triangle.bounds.max_bound[axis] > split_point;
    debug_assert(to_left || to_right ||
                 (triangle.bounds.min_bound[axis] == split_point &&
                  triangle.bounds.max_bound[axis] == split_point));

    bool on_split = !to_left && !to_right;
    if (to_left || on_split) {

      auto new_bounds = triangle.new_bounds(
          std::numeric_limits<float>::lowest(), split_point, axis);

      left_aabb = left_aabb.union_other(new_bounds);
      left_bounds.push_back(new_bounds);
      left_triangles.push_back(triangle_idx);
    }
    if (to_right) {
      auto new_bounds = triangle.new_bounds(
          split_point, std::numeric_limits<float>::max(), axis);

      right_aabb = right_aabb.union_other(new_bounds);
      right_bounds.push_back(new_bounds);
      right_triangles.push_back(triangle_idx);
    }
  }

  debug_assert(std::is_sorted(left_triangles.begin(), left_triangles.end()));
  debug_assert(std::is_sorted(right_triangles.begin(), right_triangles.end()));
  debug_assert(
      std::adjacent_find(left_triangles.begin(), left_triangles.end()) ==
      left_triangles.end());
  debug_assert(
      std::adjacent_find(right_triangles.begin(), right_triangles.end()) ==
      right_triangles.end());

  float actual_base_cost = left_triangles.size() * left_aabb.surface_area() +
                           right_triangles.size() * right_aabb.surface_area();

  // hacky floating point compare...
  debug_assert(actual_base_cost <=
               best_base_cost + std::max(1e-8, 1e-8 * best_base_cost));

  always_assert(left_triangles.size() == left_bounds.size());
  always_assert(right_triangles.size() == right_bounds.size());

  return {
      .base_cost = actual_base_cost,
      .item =
          {
              tag_v<SplitType::Spatial>,
              {
                  .left_triangles = left_triangles,
                  .right_triangles = right_triangles,
                  .left_bounds = left_bounds,
                  .right_bounds = right_bounds,
                  .left_aabb = left_aabb,
                  .right_aabb = right_aabb,
              },
          },
  };
}

template class SBVH<ExecutionModel::CPU>::Generator;
#ifndef CPU_ONLY
template class SBVH<ExecutionModel::GPU>::Generator;
#endif
} // namespace sbvh
} // namespace accel
} // namespace intersect
