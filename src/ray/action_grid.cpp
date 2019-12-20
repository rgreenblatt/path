#include "ray/action_grid.h"
#include "ray/projection_impl.h"

namespace ray {
namespace detail {
TraversalGrid::TraversalGrid(
    const TriangleProjector &projector, const scene::ShapeData *shapes,
    uint16_t num_shapes, const Eigen::Array2f &min, const Eigen::Array2f &max,
    uint16_t num_divisions_x, uint16_t num_divisions_y, bool flip_x,
    bool flip_y,
    thrust::optional<std::vector<ProjectedTriangle> *> save_triangles)
    : projector_(projector), min_(min), max_(max), min_indexes_(0, 0),
      max_indexes_(num_divisions_x, num_divisions_y), difference_(max - min),
      inverse_difference_(Eigen::Array2f(num_divisions_x, num_divisions_y) /
                          difference_),
      num_divisions_x_(num_divisions_x), num_divisions_y_(num_divisions_y),
      flip_x_(flip_x), flip_y_(flip_y),
      action_num_(num_divisions_x * num_divisions_y) {
  resize(num_shapes);
  std::vector<ProjectedTriangle> triangles;
  for (uint16_t shape_idx = 0; shape_idx < num_shapes; shape_idx++) {
    updateShape(shapes, shape_idx, triangles);
    if (save_triangles.has_value()) {
      (*save_triangles)
          ->insert((*save_triangles)->end(), triangles.begin(),
                   triangles.end());
    }
  }
}

void TraversalGrid::resize(unsigned new_num_shapes) {
  uint16_t old_num_shapes = shape_grids_.size() / num_divisions_y_;
  if (new_num_shapes > old_num_shapes) {
    shape_grids_.insert(shape_grids_.end(),
                        (new_num_shapes - shape_grids_.size()) *
                            num_divisions_y_,
                        ShapeRowPossibles());
  } else {
    shape_grids_.erase(shape_grids_.end() -
                           (shape_grids_.size() - new_num_shapes) *
                               num_divisions_y_,
                       shape_grids_.end());
  }
}

void TraversalGrid::wipeShape(unsigned shape_to_wipe) {
  std::fill_n(shape_grids_.begin() + shape_to_wipe * num_divisions_y_,
              num_divisions_y_,
              ShapeRowPossibles{num_divisions_x_, 0, num_divisions_x_, 0});
}

void TraversalGrid::updateShape(const scene::ShapeData *shapes,
                                unsigned shape_to_update,
                                std::vector<ProjectedTriangle> &triangles) {
  wipeShape(shape_to_update);

  triangles.clear();

  project_shape(shapes[shape_to_update], projector_, triangles, flip_x_,
                flip_y_);

  unsigned shape_grids_start = shape_to_update * num_divisions_y_;

  for (const auto &triangle : triangles) {
    const auto &points = triangle.points();
    auto min_p = points[0].cwiseMin(points[1].cwiseMin(points[2])).eval();
    auto max_p = points[0].cwiseMax(points[1].cwiseMax(points[2])).eval();

    // small epsilon required because of Ofast????
    auto max_grid_indexes = ((max_p - min_) * inverse_difference_ - 1e-5f)
                                .ceil()
                                .cwiseMin(max_indexes_)
                                .cwiseMax(min_indexes_)
                                .cast<uint16_t>()
                                .eval();
    auto min_grid_indexes = 
      ((min_p - min_) * inverse_difference_ + 1e-5f)
                                .floor()
                                .cwiseMin(max_indexes_)
                                .cwiseMax(min_indexes_)
                                .cast<uint16_t>()
                                .eval();

    std::array<Eigen::Array2f, 3> dirs = {
        points[1] - points[0], points[2] - points[1], points[0] - points[2]};

    thrust::optional<Eigen::Array2f> last_smaller_x;
    thrust::optional<Eigen::Array2f> last_larger_x;

    if (max_grid_indexes.y() - min_grid_indexes.y() == 1) {
      auto &shape_row_possibles =
          shape_grids_[shape_grids_start + min_grid_indexes.y()];
      shape_row_possibles[0] =
          std::min(shape_row_possibles[0], min_grid_indexes.x());
      shape_row_possibles[1] =
          std::max(shape_row_possibles[1], max_grid_indexes.x());

      continue;
    }

    for (uint16_t grid_index_y = min_grid_indexes.y();
         grid_index_y <= max_grid_indexes.y(); grid_index_y++) {

      float v =
          (float(grid_index_y) / num_divisions_y_) * difference_.y() + min_.y();
      auto get_intersection =
          [&](uint8_t point_idx) -> thrust::optional<Eigen::Array2f> {
        float t = (v - points[point_idx].y()) / dirs[point_idx].y();
        if (t > 1 || t < 0) {
          return thrust::nullopt;
        }

        return (points[point_idx] + dirs[point_idx] * t);
      };

      thrust::optional<Eigen::Array2f> smaller_x = thrust::nullopt;
      thrust::optional<Eigen::Array2f> larger_x = thrust::nullopt;

      uint8_t num_intersections = 0;
      for (uint8_t point_idx = 0; point_idx < 3; point_idx++) {
        auto out = get_intersection(point_idx);
        if (out.has_value()) {
          num_intersections++;
          if (smaller_x.has_value()) {
            // can only have 2 intersections, so must have same value in this
            // case...
            if (out->x() <= smaller_x->x()) {
              smaller_x = *out;
            } else if (out->x() >= larger_x->x()) {
              larger_x = *out;
            }
          } else {
            smaller_x = out;
            larger_x = out;
          }
        }
      }

      if (grid_index_y != min_grid_indexes.y()) {
        // TODO clean...
        Eigen::Array2f larger_smaller_x;
        Eigen::Array2f smaller_smaller_x;
        if (smaller_x.has_value()) {
          if (last_smaller_x.has_value()) {
            larger_smaller_x = smaller_x->x() > last_smaller_x->x()
                                   ? *smaller_x
                                   : *last_smaller_x;
            smaller_smaller_x = smaller_x->x() < last_smaller_x->x()
                                    ? *smaller_x
                                    : *last_smaller_x;
          } else {
            larger_smaller_x = *smaller_x;
            smaller_smaller_x = *smaller_x;
          }
        } else {
          if (last_smaller_x.has_value()) {
            larger_smaller_x = *last_smaller_x;
            smaller_smaller_x = *last_smaller_x;
          } else {
            // should very rarely occur
            continue;
          }
        }

        Eigen::Array2f smaller_larger_x;
        Eigen::Array2f larger_larger_x;
        if (larger_x.has_value()) {
          if (last_larger_x.has_value()) {
            smaller_larger_x =
                larger_x->x() < last_larger_x->x() ? *larger_x : *last_larger_x;
            larger_larger_x =
                larger_x->x() > last_larger_x->x() ? *larger_x : *last_larger_x;
          } else {
            smaller_larger_x = *larger_x;
            larger_larger_x = *larger_x;
          }
        } else {
          if (last_larger_x.has_value()) {
            smaller_larger_x = *last_larger_x;
            larger_larger_x = *last_larger_x;
          } else {
            // should very rarely occur
            continue;
          }
        }

        auto bound_clamp = [&](float bound) -> uint16_t {
          return std::clamp(bound, min_indexes_.x(), max_indexes_.x());
        };

        auto min_bound = ((smaller_smaller_x - min_) * inverse_difference_).x();
        auto max_bound = ((larger_larger_x - min_) * inverse_difference_).x();

        auto &shape_row_possibles =
            shape_grids_[shape_grids_start + (grid_index_y - 1)];

        shape_row_possibles[0] = std::min(shape_row_possibles[0],
                                          bound_clamp(std::floor(min_bound)));
        shape_row_possibles[1] =
            std::max(shape_row_possibles[1], bound_clamp(std::ceil(max_bound)));

        if (triangle.is_guaranteed()) {
          auto min_bound =
              ((larger_smaller_x - min_) * inverse_difference_).x();
          auto max_bound =
              ((smaller_larger_x - min_) * inverse_difference_).x();
          shape_row_possibles[2] = std::min(shape_row_possibles[2],
                                            bound_clamp(std::ceil(min_bound)));
          shape_row_possibles[3] = std::max(shape_row_possibles[3],
                                            bound_clamp(std::floor(max_bound)));
        }
      }

      last_smaller_x = smaller_x;
      last_larger_x = larger_x;
    }
  }
}

void TraversalGrid::copy_into(std::vector<Traversal> &traversals,
                              std::vector<Action> &actions) {
  unsigned traversal_size = num_divisions_x_ * num_divisions_y_;
  unsigned traversal_start_index = traversals.size();
  traversals.insert(traversals.end(), traversal_size, Traversal(0, 0));
  auto start_traversals = &traversals[traversal_start_index];

  for (unsigned shape_idx = 0;
       shape_idx < shape_grids_.size() / num_divisions_y_; shape_idx++) {
    for (unsigned division_y = 0; division_y < num_divisions_y_; division_y++) {
      auto &shape_row_possibles =
          shape_grids_.at(shape_idx * num_divisions_y_ + division_y);
      for (uint16_t division_x = shape_row_possibles[0];
           division_x < shape_row_possibles[1]; division_x++) {
        start_traversals[division_x + division_y * num_divisions_x_].size++;
      }
    }
  }

  unsigned action_start_index = actions.size();

  start_traversals[0].start = action_start_index;
  for (unsigned traversal_index = 1; traversal_index < traversal_size;
       traversal_index++) {
    start_traversals[traversal_index].start =
        start_traversals[traversal_index - 1].start +
        start_traversals[traversal_index - 1].size;
  }

  const auto &last_traversal = start_traversals[traversal_size - 1];

  actions.insert(actions.end(),
                 (last_traversal.start - action_start_index) +
                     last_traversal.size,
                 Action());

  std::fill(action_num_.begin(), action_num_.end(), 0);

  for (unsigned shape_idx = 0;
       shape_idx < shape_grids_.size() / num_divisions_y_; shape_idx++) {
    for (unsigned division_y = 0; division_y < num_divisions_y_; division_y++) {
      auto &shape_row_possibles =
          shape_grids_[shape_idx * num_divisions_y_ + division_y];
      for (uint16_t division_x = shape_row_possibles[0];
           division_x < shape_row_possibles[1]; division_x++) {
        unsigned traversal_index = division_y * num_divisions_x_ + division_x;
        unsigned action_idx = start_traversals[traversal_index].start +
                              action_num_[traversal_index];

        bool is_guaranteed = division_x >= shape_row_possibles[2] &&
                             division_x < shape_row_possibles[3];

        actions[action_idx] = Action(shape_idx, is_guaranteed);

        action_num_[traversal_index]++;
      }
    }
  }
}
} // namespace detail
} // namespace ray
