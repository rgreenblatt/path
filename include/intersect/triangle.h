#pragma once

#include "intersect/intersection.h"
#include "intersect/object.h"
#include "intersect/ray.h"
#include "lib/attribute.h"
#include "lib/cuda/utils.h"
#include "lib/unit_vector.h"

#include <Eigen/Geometry>

#include <array>

namespace intersect {
struct Triangle {
  std::array<Eigen::Vector3f, 3> vertices;

  ATTR_PURE_NDEBUG HOST_DEVICE inline Triangle
  transform(const Eigen::Affine3f &t) const {
    return {{t * vertices[0], t * vertices[1], t * vertices[2]}};
  }

  ATTR_PURE_NDEBUG HOST_DEVICE inline Eigen::Vector3f normal_raw() const {
    return (vertices[1] - vertices[0]).cross(vertices[2] - vertices[0]);
  }

  ATTR_PURE_NDEBUG HOST_DEVICE inline Eigen::Vector3f
  normal_scaled_by_area() const {
    return 0.5f * normal_raw();
  }

  ATTR_PURE_NDEBUG HOST_DEVICE inline UnitVector normal() const {
    return UnitVector::new_normalize(normal_raw());
  }

  ATTR_PURE_NDEBUG HOST_DEVICE inline std::array<float, 3>
  interpolation_values(const Eigen::Vector3f &point) const;

  ATTR_PURE_NDEBUG HOST_DEVICE inline accel::AABB bounds() const;

  struct InfoType {};

  ATTR_PURE_NDEBUG HOST_DEVICE inline IntersectionOp<InfoType>
  intersect(const Ray &ray) const;
};

static_assert(Object<Triangle>);
} // namespace intersect
