#pragma once

#include "lib/settings.h"

namespace rng {
namespace detail {
template <Setting S> struct RngFromSequenceGenSettings {
  S sequence_settings;
  unsigned max_sample_size = 128;

  template <typename Archive> void serialize(Archive &ar) {
    ar(NVP(sequence_settings), NVP(max_sample_size));
  }

  constexpr bool operator==(const RngFromSequenceGenSettings &) const = default;
};
} // namespace detail
} // namespace rng
