#pragma once

#include "lib/settings.h"

namespace integrate {
namespace dir_sampler {
namespace bsdf_sampler {
struct Settings : EmptySettings {};

static_assert(Setting<Settings>);
} // namespace bsdf_sampler
} // namespace dir_sampler
} // namespace integrate
