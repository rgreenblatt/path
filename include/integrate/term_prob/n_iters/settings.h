#pragma once

#include "lib/settings.h"

namespace integrate {
namespace term_prob {
namespace n_iters {
struct Settings {
  unsigned iters = 1;

  template <class Archive> void serialize(Archive &archive) {
    archive(NVP(iters));
  }

  constexpr bool operator==(const Settings &) const = default;
};

static_assert(Setting<Settings>);
} // namespace n_iters
} // namespace term_prob
} // namespace integrate