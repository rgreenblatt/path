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
  uint16_t traversal_start;
  uint8_t axis;
  float value;
  Eigen::Array2f min;
  Eigen::Array2f convert_space_coords;
  uint16_t num_divisions_x;

  HOST_DEVICE TraversalData(uint16_t traversal_start, uint8_t axis, float value,
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
                    Span<const Traversal> traversals)
      : actions(actions), traversal_data_(traversal_data),
        traversals_(traversals) {}

  HOST_DEVICE const Traversal &getCameraTraversal(unsigned group_index) const {
    return traversals_[group_index];
  }

  HOST_DEVICE const Traversal &
  getLightTraversal(unsigned light_idx, const Eigen::Vector3f &direction,
                    const Eigen::Vector3f &point) const {
    const auto &traversal_data = traversal_data_[light_idx];

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

  const Span<const Action> actions;

private:
  const Span<const TraversalData> traversal_data_;
  const Span<const Traversal> traversals_;
};
} // namespace detail
} // namespace ray
