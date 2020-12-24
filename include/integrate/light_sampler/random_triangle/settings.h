#pragma once

#include "lib/settings.h"

#include <limits>

namespace integrate {
namespace light_sampler {
namespace random_triangle {
struct Settings {
  unsigned binary_search_threshold = std::numeric_limits<unsigned>::max();

  template <class Archive> void serialize(Archive &archive) {
    archive(NVP(binary_search_threshold));
  }

  constexpr bool operator==(const Settings &) const = default;
};

static_assert(Setting<Settings>);
} // namespace random_triangle
} // namespace light_sampler
} // namespace integrate
