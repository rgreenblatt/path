#pragma once

#include "lib/settings.h"
#include "render/computation_settings.h"

namespace render {
struct GeneralSettings {
  ComputationSettings computation_settings;
  bool back_cull_emission = true;

  template <class Archive> void serialize(Archive &archive) {
    archive(CEREAL_NVP(computation_settings));
    archive(CEREAL_NVP(back_cull_emission));
  }
};

static_assert(Setting<GeneralSettings>);
} // namespace render
