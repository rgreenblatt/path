#pragma once

#include "lib/settings.h"

namespace render {
struct ComputationSettings : EmptySettings {};

static_assert(Setting<ComputationSettings>);
} // namespace render
