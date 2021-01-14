#pragma once

#include "kernel/grid_location_info.h"

namespace kernel {
struct LocationInfo {
  unsigned start_sample;
  unsigned end_sample;
  unsigned location;

  static constexpr LocationInfo
  from_grid_location_info(const GridLocationInfo &info, unsigned x_dim) {
    return {
        .start_sample = info.start_sample,
        .end_sample = info.end_sample,
        .location = info.x + info.y * x_dim,
    };
  }
};
} // namespace kernel
