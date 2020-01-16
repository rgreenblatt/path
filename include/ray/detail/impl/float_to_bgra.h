#pragma once

#include "lib/bgra.h"
#include "lib/cuda/utils.h"
#include "lib/span.h"
#include "scene/color.h"
#include "lib/printf_dbg.h"

namespace ray {
namespace detail {
inline HOST_DEVICE void float_to_bgra_impl(unsigned x, unsigned y,
                                           unsigned x_dim, unsigned y_dim,
                                           unsigned super_sampling_rate,
                                           Span<const scene::Color> colors,
                                           Span<BGRA> bgra) {
  if (x >= x_dim || y >= y_dim) {
    return;
  }

  unsigned index = x + y * x_dim;

  scene::Color color(0, 0, 0);

  unsigned effective_width = super_sampling_rate * x_dim;
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
} // namespace detail
} // namespace ray
