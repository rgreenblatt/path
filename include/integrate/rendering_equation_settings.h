#pragma once

#include "lib/settings.h"

namespace integrate {
struct RenderingEquationSettings {
  bool back_cull_emission = true;

  template <typename Archive> void serialize(Archive &archive) {
    archive(NVP(back_cull_emission));
  }

  constexpr bool operator==(const RenderingEquationSettings &) const = default;
};

static_assert(Setting<RenderingEquationSettings>);
} // namespace integrate
