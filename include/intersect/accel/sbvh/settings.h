#pragma once

#include "intersect/accel/detail/bvh/settings.h"
#include "lib/settings.h"

namespace intersect {
namespace accel {
namespace sbvh {
struct Settings {
  // alpha from original paper
  float overlap_threshold = 1e-5;
  detail::bvh::Settings bvh_settings;

  SETTING_BODY(Settings, overlap_threshold, bvh_settings);
};

static_assert(Setting<Settings>);
} // namespace sbvh
} // namespace accel
} // namespace intersect
