#pragma once

#include "lib/settings.h"

namespace rng {
namespace detail {
template <Setting S> struct RngFromSequenceGenSettings {
  S sequence_settings;
  unsigned max_sample_size = 256;

  SETTING_BODY(RngFromSequenceGenSettings, sequence_settings, max_sample_size);
};
} // namespace detail
} // namespace rng
