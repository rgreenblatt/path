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
  Eigen::Array2f min;
  Eigen::Array2f convert_space_coords;
  uint16_t num_divisions_x;

  HOST_DEVICE TraversalData(uint16_t traversal_start, uint8_t axis, float value,
                            const Eigen::Array2f &min,
                            const Eigen::Array2f &max, uint16_t num_divisions_x,
                            uint16_t num_divisions_y)
      : traversal_start(traversal_start), axis(axis), value(value), min(min),
        convert_space_coords(Eigen::Array2f(num_divisions_x, num_divisions_y) /
                             (max - min).array()),
        num_divisions_x(num_divisions_x) {}

  HOST_DEVICE TraversalData() {}

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class TraversalGrid {
public:
  // num_divisions_x > 0, num_divisions_y > 0
  TraversalGrid(const Plane &plane, const Eigen::Affine3f &transform,
                const Eigen::Projective3f &unhinging,
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
  Plane plane_;
  Eigen::Affine3f transform_;
  Eigen::Projective3f unhinging_;
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
} // namespace detail
} // namespace ray
