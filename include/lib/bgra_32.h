#pragma once

#include "lib/assert.h"
#include "lib/attribute.h"
#include "lib/cuda/utils.h"
#include "lib/eigen_wrapper.h"
#include "lib/float_rgb.h"
#include "lib/tone_map.h"

using BGRA32 = eigen_wrapper::Array4<uint8_t>;

ATTR_PURE_NDEBUG HOST_DEVICE inline BGRA32
float_rgb_to_bgra_32(const FloatRGB &float_rgb) {
  FloatRGB tone_mapped_float_rgb = tone_map(float_rgb);

  debug_assert(tone_mapped_float_rgb[0] >= 0.0f);
  debug_assert(tone_mapped_float_rgb[1] >= 0.0f);
  debug_assert(tone_mapped_float_rgb[2] >= 0.0f);
  debug_assert(tone_mapped_float_rgb[0] <= 1.0f);
  debug_assert(tone_mapped_float_rgb[1] <= 1.0f);
  debug_assert(tone_mapped_float_rgb[2] <= 1.0f);

  BGRA32 out;
  auto rgb = (tone_mapped_float_rgb * 255.0f).cast<uint8_t>().eval();
  out[0] = rgb[2];
  out[1] = rgb[1];
  out[2] = rgb[0];
  out[3] = 0;

  return out;
}
