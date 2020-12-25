#pragma once

#include "lib/settings.h"
#include "rng/rng_from_sequence_gen_settings.h"

namespace rng {
namespace sobel {
namespace detail {
struct SobelSettings : EmptySettings {};
} // namespace detail

using Settings = rng::detail::RngFromSequenceGenSettings<detail::SobelSettings>;
} // namespace sobel
} // namespace rng
