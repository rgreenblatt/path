#pragma once

#include "lib/settings.h"

#include <limits>

namespace integrate {
namespace light_sampler {
namespace random_triangle {
struct Settings {
  // / 2 needed to avoid issues with maxing out archive...
  unsigned binary_search_threshold = std::numeric_limits<unsigned>::max() / 4;

  SETTING_BODY(Settings, binary_search_threshold);
};

static_assert(Setting<Settings>);
} // namespace random_triangle
} // namespace light_sampler
} // namespace integrate
