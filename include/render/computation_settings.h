#pragma once

#include "lib/settings.h"

namespace render {
struct ComputationSettings {
  unsigned max_blocks_per_launch = 256;

  template <typename Archive> void serialize(Archive &archive) {
    archive(NVP(max_blocks_per_launch));
  }

  constexpr bool operator==(const ComputationSettings &) const = default;
};

static_assert(Setting<ComputationSettings>);
} // namespace render
