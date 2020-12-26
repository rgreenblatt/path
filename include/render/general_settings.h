#pragma once

#include "integrate/rendering_equation_settings.h"
#include "lib/settings.h"
#include "render/computation_settings.h"

namespace render {
struct GeneralSettings {
  ComputationSettings computation_settings;
  integrate::RenderingEquationSettings rendering_equation_settings;

  template <typename Archive> void serialize(Archive &archive) {
    archive(NVP(computation_settings), NVP(rendering_equation_settings));
  }

  constexpr bool operator==(const GeneralSettings &) const = default;
};

static_assert(Setting<GeneralSettings>);
} // namespace render
