#pragma once

#include "lib/settings.h"

namespace integrate {
namespace light_sampler {
namespace no_light_sampling {
struct Settings : EmptySettings {};

static_assert(Setting<Settings>);
} // namespace no_light_sampling
} // namespace light_sampler
} // namespace integrate
