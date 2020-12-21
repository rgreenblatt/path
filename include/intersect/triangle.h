#pragma once

#include "intersect/intersection.h"
#include "intersect/object.h"
#include "intersect/ray.h"
#include "lib/cuda/utils.h"

#include <Eigen/Core>

namespace intersect {
class Triangle {
public:
  HOST_DEVICE Triangle() {}

  HOST_DEVICE Triangle(std::array<Eigen::Vector3f, 3> vertices)
      : vertices_(vertices) {}

  HOST_DEVICE inline const std::array<Eigen::Vector3f, 3> &vertices() const {
    return vertices_;
  }

  HOST_DEVICE inline Triangle transform(const Eigen::Affine3f &t) const {
    return {{t * vertices_[0], t * vertices_[1], t * vertices_[2]}};
  }

  template <typename T>
  HOST_DEVICE inline T interpolate_values(const Eigen::Vector3f &point,
                                          const std::array<T, 3> &data) const;

  HOST_DEVICE inline accel::AABB bounds() const;

  using InfoType = std::tuple<>;

  HOST_DEVICE inline IntersectionOp<InfoType> intersect(const Ray &ray) const;

private:
  std::array<Eigen::Vector3f, 3> vertices_;
};

static_assert(Object<Triangle>);
} // namespace intersect
