#pragma once

#include "lib/settings.h"

namespace render {
struct WorkDivisionSettings {
  // SPEED: tune?
  // maybe max is incorrect paradigm
  // consider target and the usage of partial reductions...
  unsigned block_size = 256;
  unsigned target_x_block_size = 32;
  unsigned target_y_block_size = 8;
  unsigned max_samples_per_thread = 16;
  unsigned target_samples_per_thread = 8;

  template <typename Archive> void serialize(Archive &ar) {
    ar(NVP(block_size), NVP(target_x_block_size), NVP(target_y_block_size),
       NVP(max_samples_per_thread), NVP(target_samples_per_thread));
  }

  constexpr bool operator==(const WorkDivisionSettings &) const = default;
};

static_assert(Setting<WorkDivisionSettings>);
} // namespace render
