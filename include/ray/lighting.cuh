#pragma once

#include "lib/bgra.h"
#include "ray/cuda_ray_utils.cuh"

namespace ray {
namespace detail {
inline __host__ __device__ void float_to_bgra(unsigned x, unsigned y,
                                              unsigned width, unsigned height,
                                              unsigned super_sampling_rate,
                                              const scene::Color *colors,
                                              BGRA *bgra) {
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
  auto bounds = [](unsigned v, unsigned close) {
    return std::abs(float(v) - close) < 3;
  };
  if (bounds(x, 512) && bounds(y, 512)) {
    bgra[index] = BGRA(230, 0, 0, 0);
  } else if (bounds(x, 512) && bounds(y, 392)) {
    bgra[index] = BGRA(0, 230, 0, 0);
  } else if (bounds(x, 610) && bounds(y, 456)) {
    bgra[index] = BGRA(0, 0, 230, 0);
  }
}

__global__ void floats_to_bgras(unsigned width, unsigned height,
                                unsigned num_blocks_x, unsigned block_dim_x,
                                unsigned block_dim_y,
                                unsigned super_sampling_rate,
                                const scene::Color *colors, BGRA *bgra) {
  auto [x, y] = get_non_sparse_indexes(num_blocks_x, block_dim_x, block_dim_y);

  float_to_bgra(x, y, width, height, super_sampling_rate, colors, bgra);
}

void floats_to_bgras_cpu(unsigned width, unsigned height,
                         unsigned super_sampling_rate,
                         const scene::Color *colors, BGRA *bgra) {
  for (unsigned x = 0; x < width; x++) {
    for (unsigned y = 0; y < width; y++) {
      float_to_bgra(x, y, width, height, super_sampling_rate, colors, bgra);
    }
  }
}
} // namespace detail
} // namespace ray
