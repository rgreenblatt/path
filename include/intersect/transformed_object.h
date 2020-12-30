#pragma once

#include "intersect/object.h"
#include "intersect/ray.h"
#include "lib/attribute.h"
#include "lib/span.h"

#include <Eigen/Geometry>

namespace intersect {
class TransformedObject {
private:
  template <Object O> struct IntersectableRef {
    const O &object;
    const TransformedObject &ref;

    using InfoType = typename O::InfoType;

    ATTR_PURE_NDEBUG HOST_DEVICE inline IntersectionOp<InfoType>
    intersect(const Ray &ray) const {
      return object.intersect(ray.transform(ref.world_to_object()));
    }

    ATTR_PURE_NDEBUG HOST_DEVICE inline const accel::AABB &bounds() const {
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

  ATTR_PURE_NDEBUG HOST_DEVICE inline const Eigen::Affine3f &
  object_to_world() const {
    return object_to_world_;
  }

  ATTR_PURE_NDEBUG HOST_DEVICE inline const Eigen::Affine3f &
  world_to_object() const {
    return world_to_object_;
  }

  ATTR_PURE_NDEBUG HOST_DEVICE inline const accel::AABB &bounds() const {
    return aabb_;
  }

  template <typename O>
  ATTR_PURE_NDEBUG HOST_DEVICE inline IntersectableRef<O>
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
