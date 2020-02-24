#pragma once

#include "intersect/accel/aabb.h"
#include "intersect/impl/ray_impl.h"
#include "intersect/object.h"
#include "intersect/ray.h"
#include "intersect/triangle.h"
#include "lib/span.h"

#include <Eigen/Geometry>

namespace intersect {
class TransformedObject {
public:
  HOST_DEVICE TransformedObject() {}

  HOST_DEVICE inline unsigned idx() const { return idx_; }

  HOST_DEVICE inline const Eigen::Affine3f &object_to_world() const {
    return object_to_world_;
  }

  HOST_DEVICE inline const Eigen::Affine3f &world_to_object() const {
    return world_to_object_;
  }

  HOST_DEVICE inline const accel::AABB &aabb() const { return aabb_; }

  TransformedObject(unsigned idx, const Eigen::Affine3f &object_to_world,
                    const accel::AABB &aabb)
      : idx_(idx), object_to_world_(object_to_world),
        world_to_object_(object_to_world.inverse()), aabb_(aabb) {}

private:
  unsigned idx_;
  Eigen::Affine3f object_to_world_;
  Eigen::Affine3f world_to_object_;
  accel::AABB aabb_; // should be transformed by transform

  friend class TransformedObject;
};

template <> struct IntersectableImpl<TransformedObject> {
  template <typename First, typename... Rest>
  static HOST_DEVICE inline auto
  intersect(const Ray &ray, const TransformedObject &object_ref,
            Span<const First> first, Span<const Rest>... rest) {
    return intersect::IntersectableT<First>::intersect(
        ray.transform(object_ref.world_to_object()), first[object_ref.idx()],
        rest...);
  }
};

template <> struct BoundedImpl<TransformedObject> {
  static HOST_DEVICE inline const accel::AABB &
  bounds(const TransformedObject &object_ref) {
    return object_ref.aabb();
  }
};
} // namespace intersect
