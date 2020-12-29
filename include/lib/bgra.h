#pragma once

#include "lib/assert.h"
#include "lib/cuda/utils.h"
#include "lib/tone_map.h"

#include <Eigen/Core>

using BGRA = Eigen::Array4<uint8_t>;

HOST_DEVICE inline BGRA intensity_to_bgr(const Eigen::Array3f &intensity) {
  auto tone_mapped_intensity = tone_map(intensity);

  debug_assert(tone_mapped_intensity.x() >= 0.0f);
  debug_assert(tone_mapped_intensity.y() >= 0.0f);
  debug_assert(tone_mapped_intensity.z() >= 0.0f);
  debug_assert(tone_mapped_intensity.x() <= 1.0f);
  debug_assert(tone_mapped_intensity.y() <= 1.0f);
  debug_assert(tone_mapped_intensity.z() <= 1.0f);

  BGRA out;
  auto rgb = (tone_mapped_intensity * 255.0f).cast<uint8_t>().eval();
  out[0] = rgb[2];
  out[1] = rgb[1];
  out[2] = rgb[0];
  out[3] = 0;

  return out;
}
