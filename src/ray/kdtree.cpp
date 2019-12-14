#include "ray/kdtree.h"
#include "ray/ray_utils.h"
#include <boost/function_output_iterator.hpp>
#include <boost/geometry.hpp>
#include <boost/iterator/counting_iterator.hpp>
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
} // namespace detail
} // namespace ray
