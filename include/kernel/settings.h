#pragma once

#include "lib/settings.h"

namespace kernel {
struct Settings {
  // This is tunned to a decent extent...
  unsigned block_size = 256;
  unsigned target_x_block_size = 32;
  bool force_target_samples = false;
  unsigned forced_target_samples_per_thread = 8;
  unsigned base_num_threads = 16384;
  float samples_per_thread_scaling_power = 0.5f;
  unsigned max_samples_per_thread = 32;

  SETTING_BODY(Settings, block_size, target_x_block_size, force_target_samples,
               forced_target_samples_per_thread, base_num_threads,
               samples_per_thread_scaling_power, max_samples_per_thread);
};

static_assert(Setting<Settings>);
} // namespace kernel
