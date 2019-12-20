#pragma once

#include "ray/intersect.h"
#include "ray/projection.h"
#include "ray/ray_utils.h"
#include <lib/span.h>

namespace ray {
namespace detail {
struct Action {
  uint16_t shape_idx;
  bool is_guaranteed;

  Action(uint16_t shape_idx, bool is_guaranteed)
      : shape_idx(shape_idx), is_guaranteed(is_guaranteed) {}

  HOST_DEVICE
  Action() {}
};

struct Traversal {
  unsigned start;
  uint16_t size;

  HOST_DEVICE Traversal(unsigned start, uint16_t size)
      : start(start), size(size) {}

  HOST_DEVICE Traversal() {}
};

struct ALIGN_STRUCT(16) TraversalData {
  unsigned traversal_start;
  uint8_t axis;
  float value;
  Eigen::Array2f min;
  Eigen::Array2f convert_space_coords;
  uint16_t num_divisions_x;

  HOST_DEVICE TraversalData(unsigned traversal_start, uint8_t axis, float value,
                            const Eigen::Array2f &min,
                            const Eigen::Array2f &max, uint16_t num_divisions_x,
                            uint16_t num_divisions_y)
      : traversal_start(traversal_start), axis(axis), value(value), min(min),
        convert_space_coords(Eigen::Array2f(num_divisions_x, num_divisions_y) /
                             (1e-5f + max - min).array()),
        num_divisions_x(num_divisions_x) {}

  HOST_DEVICE TraversalData() {}

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class TraversalGrid {
public:
  // num_divisions_x > 0, num_divisions_y > 0
  TraversalGrid(const TriangleProjector &projector,
                const scene::ShapeData *shapes, uint16_t num_shapes,
                const Eigen::Array2f &min, const Eigen::Array2f &max,
                uint16_t num_divisions_x, uint16_t num_divisions_y,
                bool flip_x = false, bool flip_y = false,
                thrust::optional<std::vector<ProjectedTriangle> *>
                    save_triangles = thrust::nullopt);

  void resize(unsigned new_num_shapes);

  void wipeShape(unsigned shape_to_wipe);

  void updateShape(const scene::ShapeData *shapes, unsigned shape_to_update,
                   std::vector<ProjectedTriangle> &triangles);

  void copy_into(std::vector<Traversal> &traversals,
                 std::vector<Action> &actions);

private:
  TriangleProjector projector_;
  Eigen::Array2f min_;
  Eigen::Array2f max_;
  Eigen::Array2f min_indexes_;
  Eigen::Array2f max_indexes_;
  Eigen::Array2f difference_;
  Eigen::Array2f inverse_difference_;
  uint16_t num_divisions_x_;
  uint16_t num_divisions_y_;
  bool flip_x_;
  bool flip_y_;
  using ShapeRowPossibles = std::array<uint16_t, 4>;
  std::vector<ShapeRowPossibles> shape_grids_;
  std::vector<unsigned> action_num_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class TraversalGridsRef {
public:
  TraversalGridsRef(Span<const Action> actions,
                    Span<const TraversalData> traversal_data,
                    Span<const Traversal> traversals,
                    std::array<unsigned, 3> start_traversal_data,
                    const Eigen::Vector3f &min_bound,
                    const Eigen::Vector3f &max_bound,
                    const Eigen::Vector3f &inverse_direction_multipliers,
                    const std::array<Eigen::Array2f, 3> &min_side_bounds,
                    const std::array<Eigen::Array2f, 3> &max_side_bounds,
                    const std::array<Eigen::Array2<int>, 3> &min_side_diffs,
                    const std::array<Eigen::Array2<int>, 3> &max_side_diffs)
      : actions(actions), traversal_data_(traversal_data),
        traversals_(traversals),
        start_traversal_data_(start_traversal_data),
        min_bound_(min_bound), max_bound_(max_bound),
        inverse_direction_multipliers_({{
            get_not_axis(inverse_direction_multipliers, 0),
            get_not_axis(inverse_direction_multipliers, 1),
            get_not_axis(inverse_direction_multipliers, 2),
        }}),
        min_side_bounds_(min_side_bounds),
        max_shifted_side_bounds_({{max_side_bounds[0] - min_side_bounds[0],
                                   max_side_bounds[1] - min_side_bounds[1],
                                   max_side_bounds[2] - min_side_bounds[2]}}),
        min_side_diffs_(min_side_diffs),
        max_shifted_side_diffs_({{
            max_side_diffs[0] - min_side_diffs[0],
            max_side_diffs[1] - min_side_diffs[1],
            max_side_diffs[2] - min_side_diffs[2],
        }}),
        multipliers_({{
            max_side_diffs[0][0] - min_side_diffs_[0][0] + 1,
            max_side_diffs[1][0] - min_side_diffs_[1][0] + 1,
            max_side_diffs[2][0] - min_side_diffs_[2][0] + 1,
        }}) {}

  HOST_DEVICE const Traversal &getCameraTraversal(unsigned group_index) const {
    return traversals_[group_index];
  }

  HOST_DEVICE const Traversal &
  getTraversalFromIdx(unsigned idx, const Eigen::Vector3f &direction,
                      const Eigen::Vector3f &point) const {
    const auto &traversal_data = traversal_data_[idx];

    auto intersection = get_intersection_point(direction, traversal_data.value,
                                               point, traversal_data.axis);

    auto x_y_idx = ((intersection.array() - traversal_data.min) *
                    traversal_data.convert_space_coords)
                       .cast<unsigned>()
                       .eval();

    unsigned light_traversal_index =
        x_y_idx.x() + x_y_idx.y() * traversal_data.num_divisions_x +
        traversal_data.traversal_start;

    return traversals_[light_traversal_index];
  }

  HOST_DEVICE const Traversal &
  getGeneralTraversal(const Eigen::Vector3f &direction,
                      const Eigen::Vector3f &point) const {
    for (uint8_t axis : {0, 1, 2}) {
      const auto &inverse_direction_multiplier =
          inverse_direction_multipliers_[axis];
      const auto &min_side_bound = min_side_bounds_[axis];
      const auto &max_shifted_side_bound = max_shifted_side_bounds_[axis];
      const auto &min_side_diff = min_side_diffs_[axis];
      const auto &max_side_diff = max_shifted_side_diffs_[axis];
      const auto &multiplier = multipliers_[axis];

      auto min_intersection =
          get_intersection_point(direction, min_bound_[axis], point, axis);
      auto grid_pos_min =
          (min_intersection * inverse_direction_multiplier - min_side_bound)
              .eval();

      // could be removed...
      if (grid_pos_min[0] < 0 || grid_pos_min[1] < 0 ||
          grid_pos_min[0] > max_shifted_side_bound[0] ||
          grid_pos_min[1] > max_shifted_side_bound[1]) {
        continue;
      }

      auto max_intersection =
          get_intersection_point(direction, max_bound_[axis], point, axis);
      auto grid_pos_max =
          (max_intersection * inverse_direction_multiplier - min_side_bound)
              .eval();

      // could be removed...
      if (grid_pos_max[0] < 0 || grid_pos_max[1] < 0 ||
          grid_pos_max[0] > max_shifted_side_bound[0] ||
          grid_pos_max[1] > max_shifted_side_bound[1]) {
        continue;
      }

      auto diff =
          (grid_pos_max.cast<int>() - grid_pos_min.cast<int>() - min_side_diff)
              .eval();

      if (diff[0] < 0 || diff[1] < 0 || diff[0] > max_side_diff[0] ||
          diff[1] > max_side_diff[1]) {
        continue;
      }

      unsigned traversal_idx =
          start_traversal_data_[axis] + diff[0] + diff[1] * multiplier;

      // TODO make better...
      return getTraversalFromIdx(traversal_idx, direction, point);
    }

    return empty_traversal_;
  }

  const Span<const Action> actions;

  const Traversal empty_traversal_ = Traversal(0, 0);


private:
  const Span<const TraversalData> traversal_data_;
  const Span<const Traversal> traversals_;
  std::array<unsigned, 3> start_traversal_data_;
  Eigen::Vector3f min_bound_;
  Eigen::Vector3f max_bound_;
  std::array<Eigen::Array2f, 3> inverse_direction_multipliers_;
  std::array<Eigen::Array2f, 3> min_side_bounds_;
  std::array<Eigen::Array2f, 3> max_shifted_side_bounds_;
  std::array<Eigen::Array2<int>, 3> min_side_diffs_;
  std::array<Eigen::Array2<int>, 3> max_shifted_side_diffs_;
  std::array<int, 3> multipliers_;
};
} // namespace detail
} // namespace ray