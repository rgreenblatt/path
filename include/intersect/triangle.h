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
template <typename T> struct TriangleGen {
  std::array<Eigen::Vector3<T>, 3> vertices;

  template <typename NewT, typename F>
  ATTR_PURE_NDEBUG HOST_DEVICE inline TriangleGen<NewT> apply_gen(F &&f) const {
    return {{f(vertices[0]), f(vertices[1]), f(vertices[2])}};
  }

  template <typename NewT>
  ATTR_PURE_NDEBUG HOST_DEVICE inline TriangleGen<NewT> cast() const {
    return apply_gen<NewT>([&](const Eigen::Vector3<T> &vec) {
      return vec.template cast<NewT>();
    });
  }

  template <typename F>
  ATTR_PURE_NDEBUG HOST_DEVICE inline TriangleGen apply(F &&f) const {
    return apply_gen<T>(f);
  }

  template <typename Transform>
  ATTR_PURE_NDEBUG HOST_DEVICE inline TriangleGen
  transform(const Transform &t) const {
    return apply([&](const Eigen::Vector3<T> &vert) { return t * vert; });
  }

  ATTR_PURE_NDEBUG HOST_DEVICE inline Eigen::Vector3<T> normal_raw() const {
    return (vertices[1] - vertices[0]).cross(vertices[2] - vertices[0]);
  }

  ATTR_PURE_NDEBUG HOST_DEVICE inline Eigen::Vector3<T> centroid() const {
    return (vertices[0] + vertices[1] + vertices[2]) / 3;
  }

  ATTR_PURE_NDEBUG HOST_DEVICE inline Eigen::Vector3<T>
  normal_scaled_by_area() const {
    return 0.5 * normal_raw();
  }

  ATTR_PURE_NDEBUG HOST_DEVICE inline T area() const {
    return normal_scaled_by_area().norm();
  }

  ATTR_PURE_NDEBUG HOST_DEVICE inline UnitVectorGen<T> normal() const {
    return UnitVectorGen<T>::new_normalize(normal_raw());
  }

  ATTR_PURE_NDEBUG HOST_DEVICE inline std::array<T, 3>
  interpolation_values(const Eigen::Vector3<T> &point) const;

  // we typically just use these baryo coords
  ATTR_PURE_NDEBUG HOST_DEVICE inline std::array<T, 2>
  baryo_values(const Eigen::Vector3<T> &point) const {
    const auto arr = interpolation_values(point);
    return {arr[1], arr[2]};
  }

  ATTR_PURE_NDEBUG HOST_DEVICE inline Eigen::Vector3<T>
  value_from_baryo(const std::array<T, 2> &baryo) const {
    auto v0 = vertices[1] - vertices[0];
    auto v1 = vertices[2] - vertices[0];
    return v0 * baryo[0] + v1 * baryo[1] + vertices[0];
  }

  ATTR_PURE_NDEBUG HOST_DEVICE inline accel::AABB bounds() const;

  struct InfoType {};

  ATTR_PURE_NDEBUG HOST_DEVICE inline IntersectionOp<InfoType, T>
  intersect(const GenRay<T> &ray) const;

  static constexpr T intersect_epsilon =
      std::is_same_v<T, double> ? 1e-10 : 1e-6;
};

using Triangle = TriangleGen<float>;

static_assert(Object<Triangle>);
} // namespace intersect
