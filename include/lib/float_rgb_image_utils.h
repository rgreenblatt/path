#pragma once

#include "lib/float_rgb.h"
#include "lib/span.h"
#include "lib/tone_map.h"

#include <vector>

// naive downsample which only works for constant factors
void downsample_to(Span<const FloatRGB> float_rgb,
                   Span<FloatRGB> downsampled_float_rgb,
                   unsigned original_width, unsigned new_width) {
  always_assert(original_width % new_width == 0);

  unsigned reduction_factor = original_width / new_width;

  for (unsigned i = 0; i < new_width; ++i) {
    unsigned i_start = i * reduction_factor;
    for (unsigned j = 0; j < new_width; ++j) {
      unsigned j_start = j * reduction_factor;
      FloatRGB total = FloatRGB::Zero();
      for (unsigned i_original = i_start;
           i_original < i_start + reduction_factor; ++i_original) {
        for (unsigned j_original = j_start;
             j_original < j_start + reduction_factor; ++j_original) {
          total += float_rgb[j_original + i_original * original_width];
        }
      }
      downsampled_float_rgb[j + i * new_width] =
          total / (reduction_factor * reduction_factor);
    }
  }
}

// double for improved precision
// use tone_map for error to provide a more visually relevant metric
ATTR_PURE_NDEBUG double
compute_mean_absolute_error(Span<const FloatRGB> float_rgb,
                            Span<const FloatRGB> ground_truth, unsigned width) {

  double total = 0.f;
  for (unsigned i = 0; i < width; ++i) {
    for (unsigned j = 0; j < width; ++j) {
      unsigned idx = j + i * width;
      total += (tone_map(float_rgb[idx].template cast<double>()) -
                tone_map(ground_truth[idx].template cast<double>()))
                   .abs()
                   .sum() /
               3.;
    }
  }

  return total / (width * width);
}
