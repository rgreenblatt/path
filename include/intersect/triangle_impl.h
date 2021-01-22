#pragma once

#include "intersect/bounded.h"
#include "intersect/intersectable.h"
#include "intersect/triangle.h"

namespace intersect {
ATTR_PURE_NDEBUG HOST_DEVICE inline IntersectionOp<Triangle::InfoType>
Triangle::intersect(const Ray &ray) const {
  Eigen::Vector3f edge1 = vertices[1] - vertices[0];
  Eigen::Vector3f edge2 = vertices[2] - vertices[0];

  Eigen::Vector3f h = ray.direction->cross(edge2);
  float a = edge1.dot(h);

  constexpr float float_epsilon = 1e-6f;

  if (std::abs(a) < float_epsilon) {
    return std::nullopt;
  }
  float f = 1.f / a;
  Eigen::Vector3f s = ray.origin - vertices[0];
  float u = f * s.dot(h);
  if (u < 0.f || u > 1.f) {
    return std::nullopt;
  }
  Eigen::Vector3f q = s.cross(edge1);
  float v = f * ray.direction->dot(q);
  if (v < 0.f || u + v > 1.f) {
    return std::nullopt;
  }
  float t = f * edge2.dot(q);
  if (t > float_epsilon) {
    bool is_back_intersection = a < 0.f;
    return Intersection<InfoType>{t, is_back_intersection, InfoType{}};
  } else {
    return std::nullopt;
  }
}

ATTR_PURE_NDEBUG HOST_DEVICE inline accel::AABB Triangle::bounds() const {
  auto min_b = max_eigen_vec();
  auto max_b = min_eigen_vec();
  for (const auto &vertex : vertices) {
    min_b = min_b.cwiseMin(vertex);
    max_b = max_b.cwiseMax(vertex);
  }

  return {min_b, max_b};
}

ATTR_PURE_NDEBUG HOST_DEVICE inline std::array<float, 3>
Triangle::interpolation_values(const Eigen::Vector3f &point) const {
  Eigen::Vector3f p0 = vertices[1] - vertices[0];
  Eigen::Vector3f p1 = vertices[2] - vertices[0];
  Eigen::Vector3f p2 = point - vertices[0];
  float d00 = p0.dot(p0);
  float d01 = p0.dot(p1);
  float d11 = p1.dot(p1);
  float d20 = p2.dot(p0);
  float d21 = p2.dot(p1);
  float denom = d00 * d11 - d01 * d01;
  float v = (d11 * d20 - d01 * d21) / denom;
  float w = (d00 * d21 - d01 * d20) / denom;
  float u = 1.f - v - w;

  return {u, v, w};
}
} // namespace intersect
