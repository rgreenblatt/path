#pragma once

#include "lib/settings.h"

namespace integrate {
struct RenderingEquationSettings {
  bool back_cull_emission = true;

  SETTING_BODY(RenderingEquationSettings, back_cull_emission);
};

static_assert(Setting<RenderingEquationSettings>);
} // namespace integrate
