#pragma once

#include "lib/settings.h"

namespace integrate {
namespace term_prob {
namespace multiplier_func {
struct Settings {
  float exp = 5.0f;
  float min_prob = 0.2f;

  template <typename Archive> void serialize(Archive &archive) {
    archive(NVP(exp), NVP(min_prob));
  }

  ATTR_PURE constexpr bool operator==(const Settings &) const = default;
};

static_assert(Setting<Settings>);
} // namespace multiplier_func
} // namespace term_prob
} // namespace integrate
