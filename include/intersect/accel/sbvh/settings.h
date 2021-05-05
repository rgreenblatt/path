#pragma once

#include "intersect/accel/detail/bvh/settings.h"
#include "lib/settings.h"

namespace intersect {
namespace accel {
namespace sbvh {
struct Settings {
  // alpha from original paper
  float overlap_threshold = 1e-5;

  // spatial splits seem bad at the moment...
  bool use_spatial_splits = false;

  detail::bvh::Settings bvh_settings;

  SETTING_BODY(Settings, overlap_threshold, use_spatial_splits, bvh_settings);
};

static_assert(Setting<Settings>);
} // namespace sbvh
} // namespace accel
} // namespace intersect
