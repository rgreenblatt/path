#pragma once

#include "lib/settings.h"

namespace integrate {
namespace term_prob {
namespace constant {
struct Settings {
  float prob = 0.5f;

  template <typename Archive> void serialize(Archive &archive) {
    archive(NVP(prob));
  }

  ATTR_PURE constexpr bool operator==(const Settings &) const = default;
};

static_assert(Setting<Settings>);
} // namespace constant
} // namespace term_prob
} // namespace integrate
