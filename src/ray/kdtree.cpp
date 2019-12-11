#include "ray/kdtree.h"
#include "ray/ray_utils.h"
#include <boost/geometry.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/function_output_iterator.hpp>
#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/combine.hpp>
#include <numeric>
#include <omp.h>

#include <chrono>
#include <dbg.h>

namespace ray {
namespace detail {
using ShapeData = scene::ShapeData;

// inspired by https://www.geeksforgeeks.org/quickselect-algorithm/
size_t partition(ShapeData *shapes, std::vector<Bounds> &bounds, int axis,
                 size_t start, size_t end) {
  float x = bounds[end].center[axis];
  size_t i = start;
  for (size_t j = start; j <= end - 1; j++) {
    if (bounds[j].center[axis] <= x) {
      std::swap(bounds[i], bounds[j]);
      std::swap(shapes[i], shapes[j]);
      i++;
    }
  }
  std::swap(bounds[i], bounds[end]);
  std::swap(shapes[i], shapes[end]);

  return i;
}

// inspired by https://www.geeksforgeeks.org/quickselect-algorithm/
void kth_smallest(ShapeData *shapes, std::vector<Bounds> &bounds, int axis,
                  size_t start, size_t end, size_t k) {
  // Partition the array around last
  // element and get position of pivot
  // element in sorted array
  size_t index = partition(shapes, bounds, axis, start, end);

  // If position is same as k
  if (index - start == k || (index - start == 1 && k == 0) ||
      (end - index == 1 && k == end - start)) {
    return;
  }

  // If position is more, recur
  // for left subarray
  if (index - start > k) {
    return kth_smallest(shapes, bounds, axis, start, index - 1, k);
  }

  // Else recur for right subarray
  kth_smallest(shapes, bounds, axis, index + 1, end, k - index + start - 1);
}

constexpr int maximum_kd_depth = 25;
constexpr int final_shapes_per_division = 2;

template <bool is_min>
Eigen::Vector3f get_min_max_bound(const std::vector<Bounds> &bounds,
                                  size_t start_shape, size_t end_shape) {

  auto get_min_max = [&](const Bounds &bound) {
    return is_min ? bound.min : bound.max;
  };

  return std::accumulate(
      bounds.begin() + start_shape + 1, bounds.begin() + end_shape,
      get_min_max(bounds.at(start_shape)),
      [&](const Eigen::Vector3f &accum, const Bounds &bound) {
        return is_min ? get_min_max(bound).cwiseMin(accum)
                      : get_min_max(bound).cwiseMax(accum).eval();
      });
}

uint16_t construct_kd_tree(std::vector<KDTreeNode<AABB>> &nodes,
                           ShapeData *shapes, std::vector<Bounds> &bounds,
                           uint16_t start_shape, uint16_t end_shape,
                           int depth) {
  if (end_shape - start_shape <= final_shapes_per_division ||
      depth == maximum_kd_depth) {
    auto min_bound = get_min_max_bound<true>(bounds, start_shape, end_shape);
    auto max_bound = get_min_max_bound<false>(bounds, start_shape, end_shape);
    uint16_t index = nodes.size();
    nodes.push_back(
        KDTreeNode({start_shape, end_shape}, AABB(min_bound, max_bound)));

    return index;
  }
  const int axis = depth % 3;
  const size_t k = (end_shape - start_shape) / 2;
  kth_smallest(shapes, bounds, axis, start_shape, end_shape - 1, k);
  float median = bounds[k + start_shape].center[axis];
  int new_depth = depth + 1;
  uint16_t left_index, right_index;
  left_index = construct_kd_tree(nodes, shapes, bounds, start_shape,
                                 start_shape + k, new_depth);
  right_index = construct_kd_tree(nodes, shapes, bounds, start_shape + k,
                                  end_shape, new_depth);
  auto &left = nodes[left_index];
  auto &right = nodes[right_index];

  auto min_bounds = left.get_contents().get_min_bound().cwiseMin(
      right.get_contents().get_min_bound());
  auto max_bounds = left.get_contents().get_max_bound().cwiseMax(
      right.get_contents().get_max_bound());

  uint16_t index = nodes.size();
  nodes.push_back(KDTreeNode(KDTreeSplit(left_index, right_index, median),
                             AABB(min_bounds, max_bounds)));

  return index;
}

std::vector<KDTreeNode<AABB>> construct_kd_tree(scene::ShapeData *shapes,
                                                uint16_t num_shapes) {
  std::vector<Bounds> shape_bounds(num_shapes);
  std::transform(shapes, shapes + num_shapes, shape_bounds.begin(),
                 [](const ShapeData &shape) {
                   auto [min_bound, max_bound] = get_shape_bounds(shape);
                   Eigen::Vector3f center = shape.get_transform().translation();
                   return Bounds(min_bound, center, max_bound);
                 });

  std::vector<KDTreeNode<AABB>> nodes;

  if (num_shapes != 0) {
    construct_kd_tree(nodes, shapes, shape_bounds, 0, num_shapes, 0);
  }

  return nodes;
}

namespace bg = boost::geometry;

using Point = bg::model::point<float, 2, bg::cs::cartesian>;
using Box = bg::model::box<Point>;
using Seg = bg::model::segment<Point>;
using BoundingValue = std::pair<Box, uint16_t>;
using RTree = bg::index::rtree<BoundingValue, bg::index::rstar<16>>;

Eigen::Array2f point_to_eigen(const Point &p) {
  return Eigen::Array2f(p.get<0>(), p.get<1>());
}

void construct_traversal_tree(std::vector<Traversal> &traversals,
                              std::vector<Action> &actions,
                              std::vector<BoundingValue> &boxes_in_volume,
                              const RTree &tree, const Box &bounding,
                              uint8_t depth, uint8_t target_depth) {
  uint8_t axis = depth % 2;

  boxes_in_volume.clear();

  // metric: harmonic mean of left and right disjoint

  tree.query(bg::index::intersects(bounding),
             std::back_inserter(boxes_in_volume));

  if (boxes_in_volume.empty() || depth == target_depth) {
    uint16_t start_actions = actions.size();
    uint16_t num_actions = boxes_in_volume.size();
    for(const auto& box_index : boxes_in_volume) {
      actions.push_back(box_index.second);
    }

    traversals.push_back(Traversal(start_actions, num_actions));

    return;
  }

  uint16_t middle_index = (boxes_in_volume.size()) / 2;
#if 0
  uint16_t middle_first_half = (boxes_in_volume.size() - 1 - middle_index) / 2;
  uint16_t middle_second_half = boxes_in_volume.size() - 1 - middle_first_half;

  // some index which isn't yet tested...
  uint16_t last_tested = boxes_in_volume.size();
#endif

  float best_partition_score = -1.0f;
  float best_partition_point;
  Box best_partition_left_box;
  Box best_partition_right_box;

  Eigen::Vector2f min_bounding = point_to_eigen(bounding.min_corner());
  Eigen::Vector2f max_bounding = point_to_eigen(bounding.max_corner());

  for (uint16_t to_test :
#if 0
       {middle_first_half, middle_index, middle_second_half}
#else
       { middle_index }
#endif
  ) {
#if 0
    if (to_test == last_tested) {
      continue;
    }
#endif

    const auto &test_box = boxes_in_volume[to_test].first;

    // epsilon to avoid self intersection....
    auto min_corner_v = point_to_eigen(test_box.min_corner())[axis] - 1e-3f;
    auto max_corner_v = point_to_eigen(test_box.max_corner())[axis] + 1e-3f;

    for (float partition_point : {min_corner_v, max_corner_v}) {
      auto make_point = [&](const float other) {
        return Point(axis ? other : partition_point,
                     axis ? partition_point : other);
      };

      auto partition_min_corner = make_point(min_bounding[!axis]);
      auto partition_max_corner = make_point(max_bounding[!axis]);

      // Will this work????
      Box seg(partition_min_corner, partition_max_corner);

      Box new_left_box(bounding.min_corner(), partition_max_corner);
      Box new_right_box(partition_min_corner, bounding.max_corner());

      uint16_t num_left = 0;
      uint16_t num_right = 0;

      tree.query(bg::index::intersects(new_left_box) &&
                     !bg::index::intersects(seg),
                 boost::make_function_output_iterator(
                     [&](const auto &) { num_left++; }));
      tree.query(bg::index::intersects(new_right_box) &&
                     !bg::index::intersects(seg),
                 boost::make_function_output_iterator(
                     [&](const auto &) { num_right++; }));

      float partition_score =
          (2.0f * num_left * num_right) / (num_left + num_right);
      if (partition_score > best_partition_score) {
        best_partition_point = partition_point;
        best_partition_left_box = new_left_box;
        best_partition_right_box = new_right_box;
      }
    }
  }

  construct_traversal_tree(traversals, actions, boxes_in_volume, tree,
                           best_partition_left_box, depth + 1, target_depth);
  uint16_t left_index = traversals.size() - 1;
  construct_traversal_tree(traversals, actions, boxes_in_volume, tree,
                           best_partition_right_box, depth + 1, target_depth);
  uint16_t right_index = traversals.size() - 1;
  traversals.push_back(
      Traversal(best_partition_point, left_index, right_index));

  return;
}

std::tuple<std::vector<Traversal>, std::vector<Action>> get_traversal_grid(
    const std::vector<std::pair<ProjectedAABBInfo, uint16_t>> &shapes,
    uint8_t target_depth, std::vector<Traversal> &traversals,
    std::vector<Action> &actions) {
  auto to_point = [](const Eigen::Vector2f &v) { return Point(v.x(), v.y()); };

  std::vector<std::pair<Box, uint16_t>> boxes;

  for (const auto &shape : shapes) {
    Box b(to_point(shape.first.flattened_min),
          to_point(shape.first.flattened_max));
    boxes.push_back(std::make_pair(b, shape.second));
  }

  RTree tree(boxes);

  std::vector<BoundingValue> temp;

  construct_traversal_tree(traversals, actions, temp, tree,
                           Box(Point(-1000, -1000), Point(1000, 1000)), 0,
                           target_depth);

  return std::make_tuple(traversals, actions);
}
} // namespace detail
} // namespace ray
