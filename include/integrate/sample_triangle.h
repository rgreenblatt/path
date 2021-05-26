#pragma once

#include "intersect/triangle.h"
#include "lib/attribute.h"
#include "lib/cuda/utils.h"
#include "rng/rng.h"

namespace integrate {
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
  return triangle.baryo_to_point({s, t});
}
} // namespace integrate
