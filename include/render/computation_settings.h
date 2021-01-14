#pragma once

#include "kernel/settings.h"
#include "lib/settings.h"

namespace render {
struct ComputationSettings {
  unsigned max_blocks_per_launch = 4096;
  kernel::Settings render_work_division = {}; // TODO

  template <typename Archive> void serialize(Archive &archive) {
    archive(NVP(max_blocks_per_launch), NVP(render_work_division));
  }

  constexpr bool operator==(const ComputationSettings &) const = default;
};

static_assert(Setting<ComputationSettings>);
} // namespace render
