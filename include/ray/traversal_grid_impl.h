#pragma once

#include "ray/projection_impl.h"
#include "ray/traversal_grid.h"

namespace ray {
namespace detail {
inline HOST_DEVICE void
TraversalGrid::wipeShape(unsigned shape_to_wipe,
                         Span<ShapePossibles> shape_grids) {
  shape_grids[shape_to_wipe + start_shape_grids_] =
      ShapePossibles(Action(), num_divisions_x_, 0, num_divisions_y_, 0);
}

inline HOST_DEVICE void
TraversalGrid::updateShape(Span<const BoundingPoints> shape_bounds,
                           Span<ShapePossibles> shape_grids,
                           unsigned shape_to_update) {
  auto &shape_possibles = shape_grids[shape_to_update + start_shape_grids_];

  Eigen::Array2f min_p = Eigen::Array2f(std::numeric_limits<float>::max(),
                                        std::numeric_limits<float>::max());
  Eigen::Array2f max_p = Eigen::Array2f(std::numeric_limits<float>::lowest(),
                                        std::numeric_limits<float>::lowest());
  float min_dist = std::numeric_limits<float>::max();
  float max_dist = std::numeric_limits<float>::lowest();

  for (const auto &point : shape_bounds[shape_to_update]) {
    auto [projected_point, dist] =
        project_point(point, projector_, flip_x_, flip_y_);

    min_p = min_p.cwiseMin(projected_point);
    max_p = max_p.cwiseMax(projected_point);

    float reduced_mag = dist * min_dist_multiplier_;
    float increased_mag = dist * max_dist_multiplier_;
    // negative case:
    if (increased_mag < reduced_mag) {
      swap(reduced_mag, increased_mag);
    }

    min_dist = std::min(reduced_mag, min_dist);
    max_dist = std::max(increased_mag, max_dist);
  }

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

  shape_possibles = ShapePossibles(Action(shape_to_update, min_dist, max_dist),
                                   min_grid_indexes.x(), max_grid_indexes.x(),
                                   min_grid_indexes.y(), max_grid_indexes.y());
}

inline HOST_DEVICE void
TraversalGrid::getCount(Span<const ShapePossibles> shape_grids,
                        unsigned shape_idx, Span<int> counts) {
  const auto &shape_possible = shape_grids[shape_idx + start_shape_grids_];
  for (unsigned division_y = shape_possible.y_min;
       division_y < shape_possible.y_max; division_y++) {
    for (unsigned division_x = shape_possible.x_min;
         division_x < shape_possible.x_max; division_x++) {
      unsigned index = division_x + division_y * num_divisions_x_;
      auto location = &counts[start_count_index_ + index];
#if defined(__CUDA_ARCH__)
      atomicAdd(location, 1);
#else
      (*location)++;
#endif
    }
  }
}

inline HOST_DEVICE void
TraversalGrid::addActions(Span<const ShapePossibles> shape_grids,
                          unsigned shape_idx, Span<int> action_indexes,
                          Span<Action> actions) {
  const auto &shape_possible = shape_grids[shape_idx + start_shape_grids_];
  for (unsigned division_y = shape_possible.y_min;
       division_y < shape_possible.y_max; division_y++) {
    for (unsigned division_x = shape_possible.x_min;
         division_x < shape_possible.x_max; division_x++) {
      unsigned index = division_x + division_y * num_divisions_x_;
      auto location = &action_indexes[start_count_index_ + index];
      int action_idx;
#if defined(__CUDA_ARCH__)
      action_idx = atomicAdd(location, 1);
#else
      action_idx = (*location)++;
#endif
      actions[action_idx] = shape_possible.action;
    }
  }
}
} // namespace detail
} // namespace ray
