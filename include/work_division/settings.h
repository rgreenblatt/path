#pragma once

#include "lib/settings.h"

namespace work_division {
struct Settings {
  // This is tunned to a decent extent...
  unsigned block_size = 256;
  unsigned target_x_block_size = 32;
  bool force_target_samples = false;
  unsigned forced_target_samples_per_thread = 8;
  unsigned base_num_threads = 16384;
  float samples_per_thread_scaling_power = 0.5f;
  unsigned max_samples_per_thread = 32;

  template <typename Archive> void serialize(Archive &ar) {
    ar(NVP(block_size), NVP(target_x_block_size), NVP(force_target_samples),
       NVP(forced_target_samples_per_thread), NVP(base_num_threads),
       NVP(samples_per_thread_scaling_power), NVP(max_samples_per_thread));
  }

  ATTR_PURE constexpr bool operator==(const Settings &) const = default;
};

static_assert(Setting<Settings>);
} // namespace work_division
