#pragma once

#include "lib/settings.h"

namespace integrate {
namespace term_prob {
namespace normalize {
struct Settings {
  float min_prob = 0.1f;

  SETTING_BODY(Settings, min_prob);
};

static_assert(Setting<Settings>);
} // namespace normalize
} // namespace term_prob
} // namespace integrate
