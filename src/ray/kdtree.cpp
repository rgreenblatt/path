#include "ray/kdtree.h"
#include "ray/ray_utils.h"
#include <boost/iterator/counting_iterator.hpp>
#include <boost/range/combine.hpp>
#include <numeric>
#include <omp.h>

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
constexpr int final_shapes_per_division = 4;

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

unsigned construct_kd_tree(std::vector<KDTreeNode> &nodes, ShapeData *shapes,
                           std::vector<Bounds> &bounds, unsigned start_shape,
                           unsigned end_shape, int depth) {
  if (end_shape - start_shape <= final_shapes_per_division ||
      depth == maximum_kd_depth) {
    auto min_bound = get_min_max_bound<true>(bounds, start_shape, end_shape);
    auto max_bound = get_min_max_bound<false>(bounds, start_shape, end_shape);
    unsigned index = nodes.size();
    nodes.push_back(KDTreeNode({start_shape, end_shape}, min_bound, max_bound));

    return index;
  }
  const int axis = depth % 3;
  const size_t k = (end_shape - start_shape) / 2;
  kth_smallest(shapes, bounds, axis, start_shape, end_shape - 1, k);
  float median = bounds[k + start_shape].center[axis];
  int new_depth = depth + 1;
  unsigned left_index, right_index;
  left_index = construct_kd_tree(nodes, shapes, bounds, start_shape,
                                 start_shape + k, new_depth);
  right_index = construct_kd_tree(nodes, shapes, bounds, start_shape + k,
                                  end_shape, new_depth);
  auto &left = nodes[left_index];
  auto &right = nodes[right_index];

  auto min_bounds = left.min_bound.cwiseMin(right.min_bound);
  auto max_bounds = left.max_bound.cwiseMax(right.max_bound);

  unsigned index = nodes.size();
  nodes.push_back(
      KDTreeNode(KDTreeSplit(left_index, right_index, median), min_bounds, max_bounds));

  return index;
}

std::vector<KDTreeNode> construct_kd_tree(scene::ShapeData *shapes,
                                          unsigned num_shapes) {
  std::vector<Bounds> shape_bounds(num_shapes);
  std::transform(shapes, shapes + num_shapes, shape_bounds.begin(),
                 [](const ShapeData &shape) {
                   Eigen::Vector3f min_bound(std::numeric_limits<float>::max(),
                                             std::numeric_limits<float>::max(),
                                             std::numeric_limits<float>::max());
                   Eigen::Vector3f max_bound(
                       std::numeric_limits<float>::lowest(),
                       std::numeric_limits<float>::lowest(),
                       std::numeric_limits<float>::lowest());
                   for (auto x : {-0.5f, 0.5f}) {
                     for (auto y : {-0.5f, 0.5f}) {
                       for (auto z : {-0.5f, 0.5f}) {
                         Eigen::Vector3f transformed_edge = Eigen::Vector3f(
                             shape.get_transform() * Eigen::Vector3f(x, y, z));
                         min_bound = min_bound.cwiseMin(transformed_edge);
                         max_bound = max_bound.cwiseMax(transformed_edge);
                       }
                     }
                   }
                   // TODO check
                   Eigen::Vector3f center = shape.get_transform().translation();
                   return Bounds(min_bound, center, max_bound);
                 });

  std::vector<KDTreeNode> nodes;

  if (num_shapes != 0) {
    construct_kd_tree(nodes, shapes, shape_bounds, 0, num_shapes, 0);
  }

  return nodes;
}
} // namespace detail
} // namespace ray
