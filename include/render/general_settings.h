#pragma once

#include "integrate/rendering_equation_settings.h"
#include "lib/settings.h"
#include "render/computation_settings.h"

namespace render {
struct GeneralSettings {
  ComputationSettings computation_settings;
  integrate::RenderingEquationSettings rendering_equation_settings;

  SETTING_BODY(GeneralSettings, computation_settings,
               rendering_equation_settings);
};

static_assert(Setting<GeneralSettings>);
} // namespace render
