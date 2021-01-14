#pragma once

#include "intersect/ray.h"
#include "lib/span.h"

#include <concepts>

namespace intersectable_scene {
template <typename T>
concept RayWriter = requires(const T &t, unsigned idx,
                             const intersect::Ray &ray) {
  t.write_at(idx, ray);
};

struct SpanRayWriter {
  Span<intersect::Ray> rays;

  void write_at(unsigned idx, const intersect::Ray &ray) const {
    rays[idx] = ray;
  }
};
} // namespace intersectable_scene
