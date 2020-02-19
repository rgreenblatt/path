#pragma once

#include "intersect/accel/aabb.h"
#include "intersect/ray.h"
#include "intersect/triangle.h"
#include "lib/span.h"

#include <Eigen/Geometry>

namespace intersect {
namespace accel {
template <typename AccelMesh> class MeshInstanceRef {
public:
  HOST_DEVICE MeshInstanceRef() {}

  HOST_DEVICE inline auto operator()(const Ray &ray) {
    return (*accel_mesh_)(ray.transform(transform_));
  }

  HOST_DEVICE inline const intersect::accel::AABB &aabb() const {
    return aabb_;
  }

private:
  MeshInstanceRef(const AccelMesh *accel_mesh, const Eigen::Affine3f &transform,
                  const AABB &aabb)
      : accel_mesh_(accel_mesh), transform_(transform), aabb_(aabb) {}

  const AccelMesh *accel_mesh_;
  Eigen::Affine3f transform_;
  intersect::accel::AABB aabb_; // should be transformed by transform

  friend class MeshInstance;
};

class MeshInstance {
public:
  MeshInstance(unsigned idx, const Eigen::Affine3f &transform,
               const intersect::accel::AABB &aabb)
      : idx_(idx), transform_(transform), aabb_(aabb) {}

  HOST_DEVICE inline const intersect::accel::AABB &aabb() const {
    return aabb_;
  }

  template <typename AccelMesh>
  MeshInstanceRef<AccelMesh> get_ref(Span<const AccelMesh> accel_meshs) const {
    return {accel_meshs.data() + idx_, transform_, aabb_};
  }

private:
  unsigned idx_;
  Eigen::Affine3f transform_;
  intersect::accel::AABB aabb_; // should be transformed by transform

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
} // namespace accel
} // namespace intersect
