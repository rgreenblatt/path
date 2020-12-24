#pragma once

#include "lib/settings.h"

namespace intersectable_scene {
namespace flat_triangle {
template <Setting AccelSettings> struct Settings {
  AccelSettings accel_settings;

  template <class Archive> void serialize(Archive &archive) {
    archive(NVP(accel_settings));
  }

  constexpr bool operator==(const Settings &) const = default;
};
} // namespace flat_triangle
} // namespace intersectable_scene
