#pragma once

#include "intersect/triangle.h"

namespace intersect {

HOST_DEVICE inline IntersectionOp<std::array<unsigned, 0>>
Triangle::operator()(const Ray &ray) const {
  // mostly the same as TA code
  static constexpr float float_epsilon = 1e-4f;

  Eigen::Vector3f edge1 = vertices_[1] - vertices_[0];
  Eigen::Vector3f edge2 = vertices_[2] - vertices_[0];

  Eigen::Vector3f h = ray.direction.cross(edge2);
  float a = edge1.dot(h);

  if (std::abs(a - 0) < float_epsilon) {
    return thrust::nullopt;
  }
  float f = 1.f / a;
  Eigen::Vector3f s = ray.origin - vertices_[0];
  float u = f * s.dot(h);
  if (u < 0.f || u > 1.f) {
    return thrust::nullopt;
  }
  Eigen::Vector3f q = s.cross(edge1);
  float v = f * ray.direction.dot(q);
  if (v < 0.f || u + v > 1.f) {
    return thrust::nullopt;
  }
  float t = f * edge2.dot(q);
  if (t > float_epsilon) {
    return Intersection<std::array<unsigned, 0>>{t, std::array<unsigned, 0>{}};
  } else {
    return thrust::nullopt;
  }
}
} // namespace intersect
