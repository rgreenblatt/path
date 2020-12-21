#pragma once

#include "intersect/intersection.h"
#include "intersect/object.h"
#include "intersect/ray.h"
#include "lib/cuda/utils.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace intersect {
struct Triangle {
  std::array<Eigen::Vector3f, 3> vertices;

  HOST_DEVICE inline Triangle transform(const Eigen::Affine3f &t) const {
    return {{t * vertices[0], t * vertices[1], t * vertices[2]}};
  }

  HOST_DEVICE inline Eigen::Vector3f normal_raw() const {
    return (vertices[1] - vertices[0]).cross(vertices[2] - vertices[0]);
  }

  HOST_DEVICE inline Eigen::Vector3f normal_scaled_by_area() const {
    return 0.5f * normal_raw();
  }

  HOST_DEVICE inline Eigen::Vector3f normal() const {
    return normal_raw().normalized();
  }

  template <typename T>
  HOST_DEVICE inline T interpolate_values(const Eigen::Vector3f &point,
                                          const std::array<T, 3> &values) const;

  HOST_DEVICE inline accel::AABB bounds() const;

  struct InfoType {};

  HOST_DEVICE inline IntersectionOp<InfoType> intersect(const Ray &ray) const;
};

static_assert(Object<Triangle>);
} // namespace intersect
