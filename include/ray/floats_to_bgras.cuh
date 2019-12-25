#pragma once

#include "lib/bgra.h"
#include "ray/cuda_ray_utils.cuh"

namespace ray {
namespace detail {
inline __host__ __device__ void
float_to_bgras_impl(unsigned x, unsigned y, unsigned width, unsigned height,
                    unsigned super_sampling_rate,
                    Span<const scene::Color> colors, Span<BGRA> bgra) {
  if (x >= width || y >= height) {
    return;
  }

  unsigned index = x + y * width;

  scene::Color color(0, 0, 0);

  unsigned effective_width = super_sampling_rate * width;
  unsigned start_x = x * super_sampling_rate;
  unsigned start_y = y * super_sampling_rate;

  for (unsigned color_y = start_y; color_y < start_y + super_sampling_rate;
       color_y++) {
    for (unsigned color_x = start_x; color_x < start_x + super_sampling_rate;
         color_x++) {
      color += colors[color_x + color_y * effective_width];
    }
  }

  color /= super_sampling_rate * super_sampling_rate;

  bgra[index].head<3>() = (color * 255.0f + 0.5f)
                              .cast<int>()
                              .cwiseMax(0)
                              .cwiseMin(255)
                              .cast<uint8_t>();
}

__global__ void floats_to_bgras(unsigned width, unsigned height,
                                unsigned super_sampling_rate,
                                Span<const scene::Color> colors,
                                Span<BGRA> bgra) {
  unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned y = threadIdx.y + blockIdx.y * blockDim.y;

  float_to_bgras_impl(x, y, width, height, super_sampling_rate, colors, bgra);
}

void floats_to_bgras_cpu(unsigned width, unsigned height,
                         unsigned super_sampling_rate,
                         Span<const scene::Color> colors, Span<BGRA> bgra) {
  for (unsigned x = 0; x < width; x++) {
    for (unsigned y = 0; y < height; y++) {
      float_to_bgras_impl(x, y, width, height, super_sampling_rate, colors,
                         bgra);
    }
  }
}
} // namespace detail
} // namespace ray