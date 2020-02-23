#pragma once

#include "intersect/accel/aabb.h"
#include "intersect/object.h"
#include "intersect/ray.h"
#include "intersect/triangle.h"
#include "lib/span.h"

#include <Eigen/Geometry>

namespace intersect {
template <typename AccelTriangle> class MeshInstanceRef {
public:
  HOST_DEVICE MeshInstanceRef() {}

  HOST_DEVICE inline const AccelTriangle &accel_triangle() const {
    return *accel_triangle_;
  }

  HOST_DEVICE inline const Eigen::Affine3f &mesh_to_world() const {
    return mesh_to_world_;
  }

  HOST_DEVICE inline const Eigen::Affine3f &world_to_mesh() const {
    return world_to_mesh_;
  }

  HOST_DEVICE inline const accel::AABB &aabb() const { return aabb_; }

private:
  MeshInstanceRef(const AccelTriangle *accel_triangle,
                  const Eigen::Affine3f &mesh_to_world, const accel::AABB &aabb)
      : accel_triangle_(accel_triangle), mesh_to_world_(mesh_to_world),
        world_to_mesh_(mesh_to_world.inverse()), aabb_(aabb) {
    static_assert(intersect::Object<AccelTriangle>);
  }

  const AccelTriangle *accel_triangle_;
  Eigen::Affine3f mesh_to_world_;
  Eigen::Affine3f world_to_mesh_;
  accel::AABB aabb_; // should be transformed by transform

  friend class MeshInstance;
};

template <typename AccelTriangle>
/* requires AccelTriangle ... */
struct IntersectableImpl<MeshInstanceRef<AccelTriangle>> {
  static HOST_DEVICE inline auto
  intersect(const Ray &ray, const MeshInstanceRef<AccelTriangle> &mesh) {
    /* return 1; */
    // TODO:
    return intersect::IntersectableT<AccelTriangle>::intersect(
        ray.transform(mesh.world_to_mesh()), mesh.accel_triangle());
  }
};

template <typename AccelTriangle>
/* requires AccelTriangle ... */
struct BoundedImpl<MeshInstanceRef<AccelTriangle>> {
  static HOST_DEVICE inline const accel::AABB &
  bounds(const MeshInstanceRef<AccelTriangle> &mesh) {
    return mesh.aabb();
  }
};

class MeshInstance {
public:
  MeshInstance(unsigned idx, const Eigen::Affine3f &mesh_to_world,
               const accel::AABB &aabb)
      : idx_(idx), mesh_to_world_(mesh_to_world), aabb_(aabb) {}

  HOST_DEVICE inline unsigned idx() const { return idx_; }

  HOST_DEVICE inline const Eigen::Affine3f &mesh_to_world() const {
    return mesh_to_world_;
  }

  HOST_DEVICE inline const accel::AABB &aabb() const { return aabb_; }

  template <typename AccelMesh>
  MeshInstanceRef<AccelMesh> get_ref(Span<const AccelMesh> accel_meshs) const {
    return {accel_meshs.data() + idx_, mesh_to_world_, aabb_};
  }

private:
  unsigned idx_;
  Eigen::Affine3f mesh_to_world_;
  accel::AABB aabb_; // should be transformed by transform

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
} // namespace intersect
