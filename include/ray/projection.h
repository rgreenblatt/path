#pragma once

#include "lib/cuda_utils.h"
#include "ray/ray_utils.h"
#include "scene/shape_data.h"

#include <Eigen/Dense>

namespace ray {
namespace detail {
class ProjectedTriangle {
public:
  HOST_DEVICE const std::array<Eigen::Array2f, 3> &points() const {
    return points_;
  }

  HOST_DEVICE ProjectedTriangle(std::array<Eigen::Array2f, 3> points)
      : points_(points) {}

  HOST_DEVICE ProjectedTriangle() {}

private:
  std::array<Eigen::Array2f, 3> points_;

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

class ALIGN_STRUCT(32) TriangleProjector {
public:
  enum class Type {
    Transform,
    DirectionPlane,
  };

  TriangleProjector() {}

  TriangleProjector(const Eigen::Matrix4f &transform)
      : transform_(transform), type_(Type::Transform) {}

  TriangleProjector(const DirectionPlane &direction_plane)
      : type_(Type::DirectionPlane), direction_plane_(direction_plane) {}

  Type type() const { return type_; }

  template <typename F> HOST_DEVICE auto visit(const F &f) const {
    switch (type_) {
    case Type::DirectionPlane:
      return f(direction_plane_);
    case Type::Transform:
      return f(transform_);
    }
  }

private:
  Eigen::Matrix4f transform_;
  Type type_;
  DirectionPlane direction_plane_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

inline HOST_DEVICE Eigen::Vector3f
apply_projective(const Eigen::Vector3f &vec,
                 const Eigen::Matrix4f &projection) {
  Eigen::Vector4f homog;
  homog.template head<3>() = vec;
  homog[3] = 1.0f;
  auto out = (projection * homog).eval();

  return (out.head<3>() / out[3]).eval();
}
} // namespace detail
} // namespace ray
