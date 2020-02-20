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

template <typename T>
HOST_DEVICE inline T
Triangle::interpolate_values(const Eigen::Vector3f &point,
                             const std::array<T, 3> &data) const {
  Eigen::Vector3f p0 = vertices_[1] - vertices_[0];
  Eigen::Vector3f p1 = vertices_[2] - vertices_[0];
  Eigen::Vector3f p2 = point - vertices_[0];
  float d00 = p0.dot(p0);
  float d01 = p0.dot(p1);
  float d11 = p1.dot(p1);
  float d20 = p2.dot(p0);
  float d21 = p2.dot(p1);
  float denom = d00 * d11 - d01 * d01;
  float v = (d11 * d20 - d01 * d21) / denom;
  float w = (d00 * d21 - d01 * d20) / denom;
  float u = 1.f - v - w;

  return u * data[0] + v * data[1] + w * data[2];
}
} // namespace intersect
