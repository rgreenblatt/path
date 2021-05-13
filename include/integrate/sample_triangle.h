#pragma once

#include "intersect/triangle.h"
#include "lib/attribute.h"
#include "lib/cuda/utils.h"
#include "rng/rng.h"

namespace integrate {
ATTR_PURE_NDEBUG HOST_DEVICE inline Eigen::Vector3f
baryocentric_to_point(const intersect::Triangle &triangle, float s, float t) {
  const auto &vertices = triangle.vertices;

  // SPEED: cache vecs?
  const auto vec0 = vertices[1] - vertices[0];
  const auto vec1 = vertices[2] - vertices[0];

  return vertices[0] + vec0 * s + vec1 * t;
}

template <rng::RngState R>
[[nodiscard]] HOST_DEVICE inline std::array<float, 2>
uniform_baryocentric(R &rng) {
  float s = rng.next();
  float t = rng.next();

  if (s + t > 1.f) {
    s = 1 - s;
    t = 1 - t;
  }

  return {s, t};
}

template <rng::RngState R>
[[nodiscard]] HOST_DEVICE inline Eigen::Vector3f
sample_triangle(const intersect::Triangle &triangle, R &rng) {
  auto [s, t] = uniform_baryocentric(rng);
  return baryocentric_to_point(triangle, s, t);
}
} // namespace integrate
