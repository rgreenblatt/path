#pragma once

#include "lib/settings.h"

namespace integrate {
namespace dir_sampler {
namespace bsdf_sampler {
struct Settings {
  SETTING_BODY(Settings);
};

static_assert(Setting<Settings>);
} // namespace bsdf_sampler
} // namespace dir_sampler
} // namespace integrate
