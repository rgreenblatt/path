#include "ray/traversal_grid.h"
#include "ray/projection_impl.h"

#include <omp.h>

namespace ray {
namespace detail {
TraversalGrid::TraversalGrid(const TriangleProjector &projector,
                             const Eigen::Array2f &min,
                             const Eigen::Array2f &max, uint8_t num_divisions_x,
                             uint8_t num_divisions_y,
                             unsigned start_shape_grids,
                             unsigned start_count_index, bool flip_x,
                             bool flip_y)
    : projector_(projector), min_(min), max_(max), min_indexes_(0, 0),
      max_indexes_(num_divisions_x, num_divisions_y), difference_(max - min),
      inverse_difference_(Eigen::Array2f(num_divisions_x, num_divisions_y) /
                          difference_),
      num_divisions_x_(num_divisions_x), num_divisions_y_(num_divisions_y),
      start_shape_grids_(start_shape_grids),
      start_count_index_(start_count_index), flip_x_(flip_x), flip_y_(flip_y) {}

void TraversalGrid::wipeShape(unsigned shape_to_wipe,
                              Span<ShapePossibles> shape_grids) {
  ShapePossibles empty_v = {num_divisions_x_, 0, num_divisions_y_, 0};
  shape_grids[shape_to_wipe + start_shape_grids_] = empty_v;
}

void TraversalGrid::updateShape(Span<const scene::ShapeData> shapes,
                                Span<ShapePossibles> shape_grids,
                                unsigned shape_to_update) {
  wipeShape(shape_to_update, shape_grids);

  std::array<ProjectedTriangle, max_proj_tris> triangles;

  unsigned num_triangles = project_shape(shapes[shape_to_update], projector_,
                                         triangles, flip_x_, flip_y_);

  auto &shape_possibles = shape_grids[shape_to_update + start_shape_grids_];

  for (unsigned triangle_idx = 0; triangle_idx < num_triangles; triangle_idx++) {
    const auto &triangle = triangles[triangle_idx];
    const auto &points = triangle.points();
    auto min_p = points[0].cwiseMin(points[1].cwiseMin(points[2])).eval();
    auto max_p = points[0].cwiseMax(points[1].cwiseMax(points[2])).eval();

    // small epsilon required because of Ofast????
    auto min_grid_indexes = ((min_p - min_) * inverse_difference_ + 1e-5f)
                                .floor()
                                .cwiseMin(max_indexes_)
                                .cwiseMax(min_indexes_)
                                .cast<uint8_t>()
                                .eval();
    auto max_grid_indexes = ((max_p - min_) * inverse_difference_ - 1e-5f)
                                .ceil()
                                .cwiseMin(max_indexes_)
                                .cwiseMax(min_indexes_)
                                .cast<uint8_t>()
                                .eval();

    shape_possibles[0] = std::min(shape_possibles[0], min_grid_indexes.x());
    shape_possibles[1] = std::max(shape_possibles[1], max_grid_indexes.x());
    shape_possibles[2] = std::min(shape_possibles[2], min_grid_indexes.y());
    shape_possibles[3] = std::max(shape_possibles[3], max_grid_indexes.y());
  }
}

void TraversalGrid::getCount(Span<const ShapePossibles> shape_grids,
                             unsigned shape_idx, Span<int> counts) {
  const auto &shape_possible = shape_grids[shape_idx + start_shape_grids_];
  for (unsigned division_y = shape_possible[2]; division_y < shape_possible[3];
       division_y++) {
    for (unsigned division_x = shape_possible[0];
         division_x < shape_possible[1]; division_x++) {
      unsigned index = division_x + division_y * num_divisions_x_;
      counts[start_count_index_ + index]++;
    }
  }
}

void TraversalGrid::addActions(Span<const ShapePossibles> shape_grids,
                               unsigned shape_idx, Span<int> action_indexes,
                               Span<Action> actions) {
  const auto &shape_possible = shape_grids[shape_idx + start_shape_grids_];
  for (unsigned division_y = shape_possible[2]; division_y < shape_possible[3];
       division_y++) {
    for (unsigned division_x = shape_possible[0];
         division_x < shape_possible[1]; division_x++) {
      unsigned index = division_x + division_y * num_divisions_x_;
      int action_idx;
      action_idx = action_indexes[start_count_index_ + index];
      action_indexes[start_count_index_ + index]++;
      actions[action_idx] = Action(shape_idx);
    }
  }
}

void update_shapes(Span<TraversalGrid, false> grids,
                   Span<ShapePossibles> shape_grids,
                   Span<const scene::ShapeData, false> shapes) {
#pragma omp parallel for
  for (unsigned i = 0; i < grids.size(); i++) {
    for (unsigned shape = 0; shape < shapes.size(); shape++) {
      grids[i].updateShape(
          Span<const scene::ShapeData>(shapes.data(), shapes.size()),
          shape_grids, shape);
    }
  }
}

void update_counts(Span<TraversalGrid, false> grids,
                   Span<const ShapePossibles> shape_grids, Span<int> counts,
                   unsigned num_shapes) {
#pragma omp parallel for
  for (unsigned i = 0; i < grids.size(); i++) {
    for (unsigned shape = 0; shape < num_shapes; shape++) {
      grids[i].getCount(shape_grids, shape, counts);
    }
  }
}

void add_actions(Span<TraversalGrid, false> grids,
                 Span<const ShapePossibles> shape_grids,
                 Span<int> action_indexes, Span<Action> actions,
                 unsigned num_shapes) {
#pragma omp parallel for
  for (unsigned i = 0; i < grids.size(); i++) {
    for (unsigned shape = 0; shape < num_shapes; shape++) {
      grids[i].addActions(shape_grids, shape, action_indexes, actions);
    }
  }
}
} // namespace detail
} // namespace ray
