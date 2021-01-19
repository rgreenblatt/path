#pragma once

#include "lib/settings.h"

namespace integrate {
namespace term_prob {
namespace multiplier_func {
struct Settings {
  float exp = 50.0f;
  float min_prob = 0.03f;

  SETTING_BODY(Settings, exp, min_prob);
};

static_assert(Setting<Settings>);
} // namespace multiplier_func
} // namespace term_prob
} // namespace integrate
