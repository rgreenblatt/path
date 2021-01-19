#pragma once

#include "lib/settings.h"

namespace integrate {
namespace term_prob {
namespace constant {
struct Settings {
  float prob = 0.5f;

  SETTING_BODY(Settings, prob);
};

static_assert(Setting<Settings>);
} // namespace constant
} // namespace term_prob
} // namespace integrate
