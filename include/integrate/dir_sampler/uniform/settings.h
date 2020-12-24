#pragma once

#include "lib/settings.h"

namespace integrate {
namespace dir_sampler {
namespace uniform {
struct Settings : EmptySettings {};

static_assert(Setting<Settings>);
} // namespace uniform
} // namespace dir_sampler
} // namespace integrate
