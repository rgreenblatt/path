#pragma once

#include "intersect/bounded.h"
#include "intersect/intersectable.h"
#include "intersect/triangle.h"

namespace intersect {
template <typename T>
ATTR_PURE_NDEBUG
    HOST_DEVICE inline IntersectionOp<typename TriangleGen<T>::InfoType>
    TriangleGen<T>::intersect(const Ray &ray) const {
  Eigen::Vector3<T> edge1 = vertices[1] - vertices[0];
  Eigen::Vector3<T> edge2 = vertices[2] - vertices[0];

  Eigen::Vector3<T> h = ray.direction->cross(edge2);
  T a = edge1.dot(h);

  // TODO: float vs double...
  constexpr T float_epsilon = 1e-6;

  if (std::abs(a) < float_epsilon) {
    return std::nullopt;
  }
  T f = 1. / a;
  Eigen::Vector3<T> s = ray.origin - vertices[0];
  T u = f * s.dot(h);
  if (u < 0. || u > 1.) {
    return std::nullopt;
  }
  Eigen::Vector3<T> q = s.cross(edge1);
  T v = f * ray.direction->dot(q);
  if (v < 0. || u + v > 1.) {
    return std::nullopt;
  }
  T t = f * edge2.dot(q);
  if (t > float_epsilon) {
    bool is_back_intersection = a < 0.;
    return Intersection<InfoType>{t, is_back_intersection, InfoType{}};
  } else {
    return std::nullopt;
  }
}

template <typename T>
ATTR_PURE_NDEBUG HOST_DEVICE inline accel::AABB TriangleGen<T>::bounds() const {
  auto min_b = max_eigen_vec();
  auto max_b = min_eigen_vec();
  for (const auto &vertex : vertices) {
    min_b = min_b.cwiseMin(vertex);
    max_b = max_b.cwiseMax(vertex);
  }

  return {min_b, max_b};
}

template <typename T>
ATTR_PURE_NDEBUG HOST_DEVICE inline std::array<T, 3>
TriangleGen<T>::interpolation_values(const Eigen::Vector3<T> &point) const {
  Eigen::Vector3<T> p0 = vertices[1] - vertices[0];
  Eigen::Vector3<T> p1 = vertices[2] - vertices[0];
  Eigen::Vector3<T> p2 = point - vertices[0];
  T d00 = p0.dot(p0);
  T d01 = p0.dot(p1);
  T d11 = p1.dot(p1);
  T d20 = p2.dot(p0);
  T d21 = p2.dot(p1);
  T denom = d00 * d11 - d01 * d01;
  T v = (d11 * d20 - d01 * d21) / denom;
  T w = (d00 * d21 - d01 * d20) / denom;
  T u = 1.f - v - w;

  return {u, v, w};
}
} // namespace intersect
