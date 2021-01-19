#pragma once

#include "lib/settings.h"

namespace integrate {
namespace term_prob {
namespace n_iters {
struct Settings {
  unsigned iters = 1;

  SETTING_BODY(Settings, iters);
};

static_assert(Setting<Settings>);
} // namespace n_iters
} // namespace term_prob
} // namespace integrate
