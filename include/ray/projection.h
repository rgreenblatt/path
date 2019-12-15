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

inline Eigen::Vector3f apply_projective(const Eigen::Vector3f &vec,
                                        const Eigen::Affine3f &transform,
                                        const Eigen::Projective3f &unhinging) {
  auto first_t = transform * vec;
  /* if (std::abs(first_t.x()) > 1 || std::abs(first_t.z()) > 1) { */
  /*   return first_t; */
  /* } */
  Eigen::Vector4f homog;
  homog.template head<3>() = first_t;
  homog[3] = 1.0f;
  auto out = unhinging * homog;

  return (out.head<3>() / out[3]).eval();
}

struct Plane {
  Eigen::Vector3f normal;
  Eigen::Vector3f origin_location;

  Eigen::Array2f find_plane_coordinate(const Eigen::Vector3f &vec) const {
    auto dist = (vec.template head<2>() - origin_location.template head<2>())
                    .array()
                    .eval();
    auto ratioed =
        ((normal.template head<2>().array() * dist) / normal.z()).eval();

    return Eigen::sqrt(ratioed * ratioed + dist * dist) * Eigen::sign(dist);
  }

  Plane get_transform(const Eigen::Affine3f &transform,
                      const Eigen::Projective3f &unhinging) const {
    return Plane(transform.linear() * normal,
                 apply_projective(origin_location, transform, unhinging));
  }

  Plane(const Eigen::Vector3f &normal, const Eigen::Vector3f &origin_location)
      : normal(normal), origin_location(origin_location) {}

  Plane(uint8_t axis, float value) {
    normal = Eigen::Vector3f::Zero();
    //TODO (sgn???)
    normal[axis] = 1.0f;
    origin_location = Eigen::Vector3f::Zero();
    origin_location[axis] = value;
  }

  Plane() {}
};
} // namespace detail
} // namespace ray
