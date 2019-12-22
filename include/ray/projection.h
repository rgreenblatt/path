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

struct Plane {
  float projection_value;
  uint8_t axis;
  uint8_t first_other_axis;
  uint8_t second_other_axis;

  Plane(float projection_value, uint8_t axis)
      : projection_value(projection_value), axis(axis),
        first_other_axis((axis + 1) % 3), second_other_axis((axis + 2) % 3) {}

  Plane() {}

  template <typename T>
  inline HOST_DEVICE Eigen::Array2<T>
  get_not_axis(const Eigen::Vector3<T> &v) const {
    return Eigen::Array2<T>(v[first_other_axis], v[second_other_axis]);
  }

  template <typename T>
  inline HOST_DEVICE Eigen::Array2<T>
  get_not_axis(const Eigen::Array3<T> &v) const {
    return Eigen::Array2<T>(v[first_other_axis], v[second_other_axis]);
  }

  inline HOST_DEVICE auto
  get_intersection_point(const Eigen::Vector3f &dir,
                         const Eigen::Vector3f &pos) const {
    float dist = (projection_value - pos[axis]) / dir[axis];

    auto point = (dist * get_not_axis(dir) + get_not_axis(pos)).eval();

    return std::make_tuple(point, dist);
  }
};

struct DirectionPlane {
  Eigen::Vector3f loc_or_dir;
  Plane plane;
  bool is_loc;

  DirectionPlane(const Eigen::Vector3f &loc_or_dir, bool is_loc,
                 const Plane &plane)
      : loc_or_dir(is_loc ? loc_or_dir : loc_or_dir.normalized()), plane(plane),
        is_loc(is_loc) {}

  DirectionPlane() {}

  inline HOST_DEVICE auto
  get_intersection_point(const Eigen::Vector3f &pos) const {
    return plane.get_intersection_point(
        is_loc ? (loc_or_dir - pos).normalized().eval() : loc_or_dir, pos);
  }

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
