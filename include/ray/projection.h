#pragma once

#include "lib/cuda_utils.h"
#include "scene/shape_data.h"

#include <Eigen/Dense>

namespace ray {
namespace detail {
class ProjectedTriangle {
public:
  HOST_DEVICE bool is_guaranteed() const { return is_guaranteed_; }
  HOST_DEVICE const std::array<Eigen::Array2f, 3> &points() const {
    return points_;
  }

  HOST_DEVICE ProjectedTriangle(std::array<Eigen::Array2f, 3> points,
                                bool is_guaranteed)
      : points_(points), is_guaranteed_(is_guaranteed) {}

private:
  std::array<Eigen::Array2f, 3> points_;
  bool is_guaranteed_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct DirectionPlane {
  Eigen::Vector3f loc_or_dir;
  bool is_loc;
  float projection_value;
  uint8_t axis;

  DirectionPlane(const Eigen::Vector3f &loc_or_dir, bool is_loc,
                 float projection_value, uint8_t axis)
      : loc_or_dir(loc_or_dir), is_loc(is_loc),
        projection_value(projection_value), axis(axis) {}

  DirectionPlane() {}

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class TriangleProjector {
public:
  enum class Type {
    Transform,
    DirectionPlane,
  };

  TriangleProjector(const Eigen::Projective3f &transform)
      : type_(Type::Transform), transform_(transform) {}

  TriangleProjector(const DirectionPlane &direction_plane)
      : type_(Type::DirectionPlane), direction_plane_(direction_plane) {}

  Type type() const { return type_; }

  Eigen::Projective3f
  get_total_transform(const Eigen::Affine3f &other_transform) const {
    switch (type_) {
    case Type::DirectionPlane:
      return other_transform;
    case Type::Transform:
      return transform_ * other_transform;
    }
  }

  template <typename F> auto visit(const F &f) const {
    switch (type_) {
    case Type::DirectionPlane:
      return f(direction_plane_);
    case Type::Transform:
      return f(uint8_t());
    }
  }

private:
  Type type_;
  Eigen::Projective3f transform_;
  DirectionPlane direction_plane_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

inline Eigen::Vector3f apply_projective(const Eigen::Vector3f &vec,
                                        const Eigen::Projective3f &projection) {
  Eigen::Vector4f homog;
  homog.template head<3>() = vec;
  homog[3] = 1.0f;
  auto out = projection * homog;

  return (out.head<3>() / out[3]).eval();
}
} // namespace detail
} // namespace ray
