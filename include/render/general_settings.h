#pragma once

#include "lib/settings.h"
#include "render/computation_settings.h"

namespace render {
struct GeneralSettings {
  ComputationSettings computation_settings;
  bool back_cull_emission = true;

  template <class Archive> void serialize(Archive &archive) {
    archive(NVP(computation_settings));
    archive(NVP(back_cull_emission));
  }

  constexpr bool operator==(const GeneralSettings &) const = default;
};

static_assert(Setting<GeneralSettings>);
} // namespace render
