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

template <typename F>
std::tuple<std::vector<Traversal>, std::vector<uint8_t>, std::vector<Action>>
get_general_traversal_grid(
    const std::vector<std::pair<ProjectedAABBInfo, uint16_t>> &shapes,
    const F &get_intersection_point_for_bound, unsigned num_blocks_x,
    unsigned num_blocks_y) {
  unsigned size = num_blocks_x * num_blocks_y;

  std::vector<Traversal> traversals(size, Traversal(0, 0));
  std::vector<uint8_t> disables(size, false);
  std::vector<Action> actions;

  using point = bg::model::point<float, 2, bg::cs::cartesian>;
  using box = bg::model::box<point>;
  using bounding_value = std::pair<box, uint16_t>;

  auto to_point = [](const Eigen::Vector2f &v) { return point(v.x(), v.y()); };

  std::vector<std::pair<box, uint16_t>> boxes;

  for (const auto &shape : shapes) {
    box b(to_point(shape.first.flattened_min),
          to_point(shape.first.flattened_max));
    boxes.push_back(std::make_pair(b, shape.second));
  }

  bg::index::rtree<bounding_value, bg::index::rstar<16>> tree(boxes);

  std::vector<uint16_t> shape_indexes;

  for (unsigned block_index_y = 0; block_index_y < num_blocks_y;
       block_index_y++) {
    for (unsigned block_index_x = 0; block_index_x < num_blocks_x;
         block_index_x++) {
      auto get_intersection_point = [&](bool is_x_min, bool is_y_min) {
        return get_intersection_point_for_bound(
            is_x_min ? block_index_x : block_index_x + 1,
            is_y_min ? block_index_y : block_index_y + 1);
      };

      auto l_l = get_intersection_point(true, true);
      auto h_l = get_intersection_point(false, true);
      auto l_h = get_intersection_point(true, false);
      auto h_h = get_intersection_point(false, false);

#if 0
      bg::model::polygon<point> poly{
          {to_point(l_l), to_point(h_l), to_point(l_h), to_point(h_h)}, {}};

      bg::correct(poly);
#else
      box poly(to_point(l_l.cwiseMin(h_l).cwiseMin(l_h).cwiseMin(h_h)),
               to_point(l_l.cwiseMax(h_l).cwiseMax(l_h).cwiseMax(h_h)));
#endif

      shape_indexes.clear();

      tree.query(
          bg::index::intersects(poly),
          boost::make_function_output_iterator([&](const bounding_value &v) {
            shape_indexes.push_back(v.second);
          }));

      unsigned index = block_index_x + block_index_y * num_blocks_x;

      if (shape_indexes.empty()) {
        disables[index] = true;
        continue;
      }

      auto is_same_traversal = [&](const unsigned index) {
        const auto &other = traversals[index];
        const auto &end = actions.begin() + other.start + other.size;

        bool valid = !disables[index] && other.size == shape_indexes.size();

        unsigned i = 0;
        for (auto iter = actions.begin() + other.start; valid && iter != end;
             iter++) {
          const auto &other_traversal = *iter;
          valid = valid && other_traversal.shape_idx == shape_indexes[i];

          i++;
        }

        return valid;
      };

      if (block_index_x != 0) {
        unsigned left_index = block_index_x - 1 + block_index_y * num_blocks_x;
        if (is_same_traversal(left_index)) {
          traversals[index] = traversals[left_index];
          continue;
        }
      }

#if 1
      if (block_index_y != 0) {
        unsigned above_index =
            block_index_x + (block_index_y - 1) * num_blocks_x;
        if (is_same_traversal(above_index)) {
          traversals[index] = traversals[above_index];
          continue;
        }
      }

      if (block_index_y != 0 && block_index_x < num_blocks_x - 1) {
        unsigned above_right_index =
            block_index_x + 1 + (block_index_y - 1) * num_blocks_x;
        if (is_same_traversal(above_right_index)) {
          traversals[index] = traversals[above_right_index];
          continue;
        }
      }
#endif

      traversals[index] =
          Traversal(actions.size(), actions.size() + shape_indexes.size());

      for (auto idx : shape_indexes) {
        actions.push_back(Action(idx));
      }
    }
  }

  return std::make_tuple(traversals, disables, actions);
}

std::tuple<std::vector<Traversal>, std::vector<uint8_t>, std::vector<Action>>
get_traversal_grid_from_transform(
    const std::vector<std::pair<ProjectedAABBInfo, uint16_t>> &shapes,
    unsigned width, unsigned height, const scene::Transform &transform_v,
    unsigned block_dim_x, unsigned block_dim_y, unsigned num_blocks_x,
    unsigned num_blocks_y, uint8_t axis, float value_to_project_to) {

  const auto &world_space_eye = transform_v.translation();

  return get_general_traversal_grid(
      shapes,
      [&](unsigned block_idx_x, unsigned block_idx_y) {
        auto dir = initial_world_space_direction(
            block_idx_x * block_dim_x, block_idx_y * block_dim_y, width, height,
            world_space_eye, transform_v);

        return get_intersection_point(dir, value_to_project_to, world_space_eye,
                                      axis);
      },
      num_blocks_x, num_blocks_y);
}

std::tuple<std::vector<Traversal>, std::vector<uint8_t>, std::vector<Action>>
get_traversal_grid_from_bounds(
    const std::vector<std::pair<ProjectedAABBInfo, uint16_t>> &shapes,
    const Eigen::Array2f &min_bound, const Eigen::Array2f &max_bound,
    unsigned num_blocks_x, unsigned num_blocks_y) {

  return get_general_traversal_grid(
      shapes,
      [&](unsigned x, unsigned y) {
        Eigen::Array2f interp(x, y);
        interp.array() /= Eigen::Array2f(num_blocks_x, num_blocks_y);

        auto out = (max_bound * interp + min_bound * (1.0f - interp)).eval();

        return out;
      },
      num_blocks_x, num_blocks_y);
}
} // namespace detail
} // namespace ray
