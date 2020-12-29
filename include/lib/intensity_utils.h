#pragma once

#include "lib/span.h"
#include "lib/tone_map.h"

#include <Eigen/Core>

#include <vector>

// naive downsample which only works for constant factors
void downsample_to(Span<const Eigen::Array3f> intensities,
                   Span<Eigen::Array3f> downsampled_intensities,
                   unsigned original_width, unsigned new_width) {
  always_assert(original_width % new_width == 0);

  unsigned reduction_factor = original_width / new_width;

  for (unsigned i = 0; i < new_width; ++i) {
    unsigned i_start = i * reduction_factor;
    for (unsigned j = 0; j < new_width; ++j) {
      unsigned j_start = j * reduction_factor;
      Eigen::Array3f total = Eigen::Array3f::Zero();
      for (unsigned i_original = i_start;
           i_original < i_start + reduction_factor; ++i_original) {
        for (unsigned j_original = j_start;
             j_original < j_start + reduction_factor; ++j_original) {
          total += intensities[j_original + i_original * original_width];
        }
      }
      downsampled_intensities[j + i * new_width] =
          total / (reduction_factor * reduction_factor);
    }
  }
}

// double for improved precision
// use tone_map for error to provide a more visually relevant metric
double compute_mean_absolute_error(Span<const Eigen::Array3f> intensities,
                                   Span<const Eigen::Array3f> ground_truth,
                                   unsigned width) {

  double total = 0.f;
  for (unsigned i = 0; i < width; ++i) {
    for (unsigned j = 0; j < width; ++j) {
      unsigned idx = j + i * width;
      total += (tone_map(intensities[idx].template cast<double>()) -
                tone_map(ground_truth[idx].template cast<double>()))
                   .abs()
                   .sum() /
               3.;
    }
  }

  return total / (width * width);
}
