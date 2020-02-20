#pragma once

#include "lib/cuda/utils.h"

#include <Eigen/Core>

using RGBA = Eigen::Array4<uint8_t>;

HOST_DEVICE inline RGBA intensity_to_rgb(const Eigen::Vector3f &intensity) {
  assert(intensity.x() >= 0.0f);
  assert(intensity.y() >= 0.0f);
  assert(intensity.z() >= 0.0f);
  assert(intensity.x() <= 1.0f);
  assert(intensity.y() <= 1.0f);
  assert(intensity.z() <= 1.0f);

  RGBA out;
  out.head<3>() = (intensity * 255.0f).cast<uint8_t>();
  out[3] = 0;

  return out;
}
