#pragma once

#include "lib/cuda/utils.h"

#include <Eigen/Geometry>

namespace material {
HOST_DEVICE inline Eigen::Vector3f
reflect_over_normal(const Eigen::Vector3f &vec, const Eigen::Vector3f &normal) {
  return (vec + 2.0f * -vec.dot(normal) * normal).normalized();
};

HOST_DEVICE inline Eigen::Vector3f
refract_by_normal(float ior, const Eigen::Vector3f &vec,
                  const Eigen::Vector3f &normal) {
  float cos_to_normal = -vec.dot(normal);

  bool exiting_media = cos_to_normal < 0;

  float frac = exiting_media ? ior : 1 / ior;

  float in_sqrt = 1 - frac * frac * (1 - cos_to_normal * cos_to_normal);

  Eigen::Vector3f effective_normal = exiting_media ? -normal : normal;

  if (in_sqrt > 0) {
    return (Eigen::AngleAxisf(std::acos(std::sqrt(in_sqrt)),
                              vec.cross(effective_normal).normalized()) *
            -effective_normal)
        .normalized();
  } else {
    return reflect_over_normal(vec, normal);
  }
};
} // namespace material
