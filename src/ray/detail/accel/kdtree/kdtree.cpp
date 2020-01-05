#include "ray/detail/accel/kdtree/kdtree.h"

#include <boost/function_output_iterator.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/combine.hpp>

#include <numeric>

namespace ray {
namespace detail {
namespace accel {
namespace kdtree {
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

unsigned construct_kd_tree(std::vector<KDTreeNode<AABB>> &nodes,
                           ShapeData *shapes, std::vector<Bounds> &bounds,
                           unsigned start_shape, unsigned end_shape,
                           unsigned depth, unsigned target_depth,
                           unsigned final_shapes_per) {
  if (end_shape - start_shape <= final_shapes_per || depth == target_depth) {
    auto min_bound = get_min_max_bound<true>(bounds, start_shape, end_shape);
    auto max_bound = get_min_max_bound<false>(bounds, start_shape, end_shape);
    unsigned index = nodes.size();
    nodes.push_back(
        KDTreeNode({start_shape, end_shape}, AABB(min_bound, max_bound)));

    return index;
  }
  const unsigned axis = depth % 3;
  const size_t k = (end_shape - start_shape) / 2;
  kth_smallest(shapes, bounds, axis, start_shape, end_shape - 1, k);
  float median = bounds[k + start_shape].center[axis];
  unsigned new_depth = depth + 1;
  unsigned left_index, right_index;
  left_index =
      construct_kd_tree(nodes, shapes, bounds, start_shape, start_shape + k,
                        new_depth, target_depth, final_shapes_per);
  right_index =
      construct_kd_tree(nodes, shapes, bounds, start_shape + k, end_shape,
                        new_depth, target_depth, final_shapes_per);
  auto &left = nodes[left_index];
  auto &right = nodes[right_index];

  auto min_bounds = left.get_contents().get_min_bound().cwiseMin(
      right.get_contents().get_min_bound());
  auto max_bounds = left.get_contents().get_max_bound().cwiseMax(
      right.get_contents().get_max_bound());

  unsigned index = nodes.size();
  nodes.push_back(KDTreeNode(KDTreeSplit(left_index, right_index, median),
                             AABB(min_bounds, max_bounds)));

  return index;
}

std::vector<KDTreeNode<AABB>> construct_kd_tree(scene::ShapeData *shapes,
                                                unsigned num_shapes,
                                                unsigned target_depth,
                                                unsigned target_shapes_per) {
  std::vector<Bounds> shape_bounds(num_shapes);
  std::transform(shapes, shapes + num_shapes, shape_bounds.begin(),
                 [](const ShapeData &shape) {
                   auto [min_bound, max_bound] =
                       get_shape_bounds(shape.get_transform());
                   Eigen::Vector3f center = shape.get_transform().translation();
                   return Bounds(min_bound, center, max_bound);
                 });

  std::vector<KDTreeNode<AABB>> nodes;

  if (num_shapes != 0) {
    construct_kd_tree(nodes, shapes, shape_bounds, 0, num_shapes, 0,
                      target_depth, target_shapes_per);
  }

  return nodes;
}
} // namespace kdtree
} // namespace accel
} // namespace detail
} // namespace ray
