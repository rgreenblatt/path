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
private:
  template <Object O> struct IntersectableRef {
    const O &object;
    const TransformedObject &ref;

    using InfoType = typename O::InfoType;

    HOST_DEVICE inline IntersectionOp<InfoType>
    intersect(const Ray &ray) const {
      return object.intersect(ray.transform(ref.world_to_object()));
    }

    HOST_DEVICE inline const accel::AABB &bounds() const {
      return ref.bounds();
    }
  };

  static_assert(Object<IntersectableRef<MockObject>>);

public:
  HOST_DEVICE TransformedObject() {}

  // this object should have the same AABB as the one used to generate the
  // IntersectableRef
  template <Bounded O>
  TransformedObject(const Eigen::Affine3f &object_to_world, const O &object)
      : object_to_world_(object_to_world),
        world_to_object_(object_to_world.inverse()),
        aabb_(object.bounds().transform(object_to_world)) {}

  HOST_DEVICE inline const Eigen::Affine3f &object_to_world() const {
    return object_to_world_;
  }

  HOST_DEVICE inline const Eigen::Affine3f &world_to_object() const {
    return world_to_object_;
  }

  HOST_DEVICE inline const accel::AABB &bounds() const { return aabb_; }

  template <typename O>
  HOST_DEVICE inline IntersectableRef<O>
  get_intersectable(const O &object) const {
    return {object, *this};
  }

private:
  Eigen::Affine3f object_to_world_;
  Eigen::Affine3f world_to_object_;
  accel::AABB aabb_;
};

static_assert(Bounded<TransformedObject>);
} // namespace intersect
