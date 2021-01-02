#pragma once

#include "intersect/ray.h"

#include <concepts>

namespace intersectable_scene {
template <typename T>
concept RayWriter = requires(T &t, unsigned idx, const intersect::Ray &ray) {
  t.write_at(idx, ray);
};
} // namespace intersectable_scene
