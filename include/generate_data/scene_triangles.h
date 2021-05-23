#pragma once

#include "intersect/triangle.h"
#include "lib/attribute.h"

#include <Eigen/Geometry>

namespace generate_data {
template <typename T> struct SceneTrianglesGen {
  intersect::TriangleGen<T> triangle_onto;
  intersect::TriangleGen<T> triangle_blocking;
  intersect::TriangleGen<T> triangle_light;

  template <typename NewT, typename F>
  ATTR_PURE_NDEBUG inline SceneTrianglesGen<NewT> apply_gen(F &&f) const {
    return {
        .triangle_onto = f(triangle_onto),
        .triangle_blocking = f(triangle_blocking),
        .triangle_light = f(triangle_light),
    };
  }

  template <typename NewT>
  ATTR_PURE_NDEBUG HOST_DEVICE inline SceneTrianglesGen<NewT> cast() const {
    return apply_gen<NewT>([&](const intersect::TriangleGen<T> &tri) {
      return tri.template cast<NewT>();
    });
  }

  template <typename F>
  ATTR_PURE_NDEBUG inline SceneTrianglesGen apply(F &&f) const {
    return apply_gen<T>(f);
  }

  template <typename Transform>
  ATTR_PURE_NDEBUG inline SceneTrianglesGen
  apply_transform(const Transform &transform) const {
    return apply([&](const intersect::TriangleGen<T> &tri) {
      return tri.transform(transform);
    });
  }
};

using SceneTriangles = SceneTrianglesGen<double>;
} // namespace generate_data
