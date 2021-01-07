#pragma once

#include "work_division/grid_location_info.h"

namespace work_division {
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
} // namespace work_division
