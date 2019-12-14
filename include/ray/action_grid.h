#pragma once

#include "ray/projection.h"
#include "ray/ray_utils.h"

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

struct TraversalData {
  uint16_t traversal_start;
  uint8_t axis;
  float value;

  HOST_DEVICE TraversalData(uint16_t traversal_start, uint8_t axis, float value)
      : traversal_start(traversal_start), axis(axis), value(value) {}

  HOST_DEVICE TraversalData() {}
};

class TraversalGrid {
public:
  // num_divisions_x > 0, num_divisions_y > 0
  TraversalGrid(const Plane &plane, const Eigen::Projective3f &transform,
                const scene::ShapeData *shapes, uint16_t num_shapes,
                const Eigen::Array2f &min, const Eigen::Array2f &max,
                uint8_t num_divisions_x, uint8_t num_divisions_y);

  void resize(unsigned new_num_shapes);

  void wipeShape(unsigned shape_to_wipe);

  void updateShape(const scene::ShapeData *shapes, unsigned shape_to_update);

  void copy_into(std::vector<Traversal> &traversals,
                 std::vector<Action> &actions);

private:
  Plane plane_;
  Eigen::Projective3f transform_;
  Eigen::Array2f min_;
  Eigen::Array2f max_;
  Eigen::Array2f difference_;
  Eigen::Array2f inverse_difference_;
  uint8_t num_divisions_x_;
  uint8_t num_divisions_y_;
  using ShapeRowPossibles = std::array<uint8_t, 4>;
  std::vector<ShapeRowPossibles> shape_grids_;
  std::vector<ProjectedTriangle> triangles_;
  std::vector<unsigned> action_num_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
} // namespace detail
} // namespace ray
