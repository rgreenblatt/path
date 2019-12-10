#include "ray/kdtree.h"
#include "ray/ray_utils.h"
#include <boost/iterator/counting_iterator.hpp>
#include <boost/range/combine.hpp>
#include <numeric>
#include <omp.h>
#include <boost/geometry.hpp>
#include <boost/range/adaptor/indexed.hpp>

#include <dbg.h>
#include <chrono>

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

  std::vector<KDTreeNode<AABB>> nodes;

  if (num_shapes != 0) {
    construct_kd_tree(nodes, shapes, shape_bounds, 0, num_shapes, 0);
  }

  return nodes;
}

#if 0
thrust::optional<std::array<uint16_t, 2>> get_traversal_grid(
    const std::vector<KDTreeNode<ProjectedAABBInfo>> &nodes, unsigned width,
    unsigned height, const scene::Transform &m_film_to_world,
    unsigned block_size_x, unsigned block_size_y, uint8_t axis,
    float value_to_project_to, std::vector<Traversal> &traversals,
    std::vector<uint8_t> &disables, std::vector<Action> &actions,
    unsigned node_index, unsigned start_block_index_x,
    unsigned end_block_index_x, unsigned start_block_index_y,
    unsigned end_block_index_y, unsigned num_blocks_x) {
  auto &world_space_eye = m_film_to_world.translation();
  auto get_intersection_point = [&](bool is_x_min, bool is_y_min) {
    auto dir = initial_world_space_direction(
        is_x_min ? start_block_index_x * block_size_x
                 : end_block_index_x * block_size_x - 1,
        is_y_min ? start_block_index_y * block_size_y
                 : end_block_index_y * block_size_y - 1,
        width, height, m_film_to_world.translation(), m_film_to_world);
    float dist = (value_to_project_to - world_space_eye[axis]) / dir[axis];

    return dist * get_not_axis(dir, axis) + get_not_axis(world_space_eye, axis);
  };

  auto l_l = get_intersection_point(true, true);
  auto h_l = get_intersection_point(false, true);
  auto l_h = get_intersection_point(true, false);
  auto h_h = get_intersection_point(false, false);

#if 0
  const Eigen::Vector2f min_intersection =
      l_l.cwiseMin(h_l).cwiseMin(l_h).cwiseMin(h_h);
  const Eigen::Vector2f max_intersection =
      l_l.cwiseMax(h_l).cwiseMax(l_h).cwiseMax(h_h);

  auto min_intersection_dist =
      nodes[node_index].get_contents().getInsideDist(min_intersection);
  auto max_intersection_dist =
      nodes[node_index].get_contents().getInsideDist(max_intersection);
#endif

  const auto& node = nodes[node_index];
  const auto& node_contents = node.get_contents();

  auto l_l_intersection_dist = node_contents.getInsideDist(l_l);
  auto h_l_intersection_dist = node_contents.getInsideDist(h_l);
  auto l_h_intersection_dist = node_contents.getInsideDist(l_h);
  auto h_h_intersection_dist = node_contents.getInsideDist(h_h);

  auto set_traversals_same = [&](const Traversal &traversal, bool disable) {
    for (unsigned block_index_x = start_block_index_x;
         block_index_x < end_block_index_x; block_index_x++) {
      for (unsigned block_index_y = start_block_index_y;
           block_index_y < end_block_index_y; block_index_y++) {
        unsigned index = block_index_x + block_index_y * num_blocks_x;
        traversals[index] = traversal;
        disables[index] = disable;
      }
    }
  };

  if (l_l_intersection_dist.has_value() || h_l_intersection_dist.has_value() ||
      l_h_intersection_dist.has_value() || h_h_intersection_dist.has_value()) {
    if (l_l_intersection_dist.has_value() &&
        l_h_intersection_dist.has_value() &&
        h_l_intersection_dist.has_value() && h_h_intersection_dist.has_value()
#if 0
        &&
        l_l_intersection_dist->type == l_h_intersection_dist->type &&
        l_l_intersection_dist->type == h_l_intersection_dist->type &&
        l_l_intersection_dist->type == h_h_intersection_dist->type
#endif
    ) {
      thrust::optional<std::array<uint16_t, 2>> out;

      node.case_split_or_data(
          [&](const KDTreeSplit &split) {
            auto get_result = [&](uint16_t node_index) {
              return get_traversal_grid(
                  nodes, width, height, m_film_to_world, block_size_x,
                  block_size_y, axis, value_to_project_to, traversals, disables,
                  actions, node_index, start_block_index_x, end_block_index_x,
                  start_block_index_y, end_block_index_y, num_blocks_x);
            };
            auto left_result = get_result(split.left_index);
            auto right_result = get_result(split.right_index);
            
            if (left_result.has_value()) {

            }
          },
          [&](const std::array<uint16_t, 2> &data) { out = data; 
          // push back action and increment traversals
          //
          });

      set_traversals_same(Traversal(0, 10), false);
    }
  } else {
    return std::array<uint16_t, 2>{0, 0};
  }
}
#endif

namespace bg = boost::geometry;

template <typename F>
std::tuple<std::vector<Traversal>, std::vector<uint8_t>, std::vector<Action>>
get_general_traversal_grid(
    const std::vector<KDTreeNode<ProjectedAABBInfo>> &nodes,
    const F &get_intersection_point_for_bound, unsigned num_blocks_x, unsigned num_blocks_y) {
  unsigned size = num_blocks_x * num_blocks_y;

  std::vector<Traversal> traversals(size, Traversal(0, 0));
  std::vector<uint8_t> disables(size, false);
  std::vector<Action> actions;

  using point = bg::model::point<float, 2, bg::cs::cartesian>;
  using box = bg::model::box<point>;
  using bounding_value = std::pair<box, uint16_t>;

  auto to_point = [](const Eigen::Vector2f &v) { return point(v.x(), v.y()); };

  std::vector<std::pair<ProjectedAABBInfo, std::array<uint16_t, 2>>>
      aa_bb_around_data;
  std::vector<std::pair<box, uint16_t>> boxes;

  for (const auto &node : nodes) {
    node.case_split_or_data(
        [](const auto &) {},
        [&](const std::array<uint16_t, 2> data) {
          const auto &contents = node.get_contents();
          box b(to_point(contents.flattened_min),
                to_point(contents.flattened_max));
          boxes.push_back(std::make_pair(b, aa_bb_around_data.size()));
          aa_bb_around_data.push_back(std::make_pair(contents, data));
        });
  }
  
  bg::index::rtree<bounding_value, bg::index::rstar<16>> tree(boxes);

  std::vector<bounding_value> aa_bb_indexes;

  for (unsigned block_index_y = 0; block_index_y < num_blocks_y;
       block_index_y++) {
    for (unsigned block_index_x = 0; block_index_x < num_blocks_x;
         block_index_x++) {
      auto get_intersection_point = [&](bool is_x_min, bool is_y_min) {
        return get_intersection_point_for_bound(
            is_x_min ? block_index_x : block_index_x + 1,
            is_y_min ? block_index_y : block_index_y + 1);
#if 0
        auto dir = initial_world_space_direction(
            is_x_min ? block_index_x * block_dim_x
                     : (block_index_x + 1) * block_dim_x - 1,
            is_y_min ? block_index_y * block_dim_y
                     : (block_index_y + 1) * block_dim_y - 1,
            width, height, m_film_to_world.translation(), m_film_to_world);

        float dist = (value_to_project_to - world_space_eye[axis]) / dir[axis];

        return (dist * get_not_axis(dir, axis) +
                get_not_axis(world_space_eye, axis))
            .eval();
#endif
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

      aa_bb_indexes.clear();

      tree.query(bg::index::intersects(poly), std::back_inserter(aa_bb_indexes));

      unsigned index = block_index_x + block_index_y * num_blocks_x;

      if (aa_bb_indexes.empty()) {
        disables[index] = true;
        continue;
      }

      auto is_same_traversal = [&](const unsigned index) {
        const auto &other = traversals[index];
        const auto &end = actions.begin() + other.start + other.size;

        bool valid = !disables[index] && other.size == aa_bb_indexes.size();

        unsigned i = 0;
        for (auto iter = actions.begin() + other.start; valid && iter != end;
             iter++) {
          const auto &other_traversal = *iter;
          const auto &corresponding_data =
              aa_bb_around_data[aa_bb_indexes[i].second].second;

          valid = valid &&
                  other_traversal.shape_idx_start == corresponding_data[0] &&
                  other_traversal.shape_idx_start == corresponding_data[0];

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

      traversals[index] =
          Traversal(actions.size(), actions.size() + aa_bb_indexes.size());

      for (auto idx : aa_bb_indexes) {
        const auto &data = aa_bb_around_data[idx.second].second;
        actions.push_back(Action(data[0], data[1]));
      }
    }
  }

  return std::make_tuple(traversals, disables, actions);
}

std::tuple<std::vector<Traversal>, std::vector<uint8_t>, std::vector<Action>>
get_traversal_grid_from_transform(
    const std::vector<KDTreeNode<ProjectedAABBInfo>> &nodes, unsigned width,
    unsigned height, const scene::Transform &transform_v, unsigned block_dim_x,
    unsigned block_dim_y, unsigned num_blocks_x, unsigned num_blocks_y,
    uint8_t axis, float value_to_project_to) {

  const auto &world_space_eye = transform_v.translation();

  return get_general_traversal_grid(
      nodes,
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
    const std::vector<KDTreeNode<ProjectedAABBInfo>> &nodes,
    const Eigen::Array2f &min_bound, const Eigen::Array2f &max_bound,
    unsigned num_blocks_x, unsigned num_blocks_y) {

  return get_general_traversal_grid(
      nodes,
      [&](unsigned x, unsigned y) {
        Eigen::Array2f interp(x, y);
        interp.array() /= Eigen::Array2f(num_blocks_x, num_blocks_y);

        return (max_bound * interp + min_bound).eval();
      },
      num_blocks_x, num_blocks_y);
}
} // namespace detail
} // namespace ray
