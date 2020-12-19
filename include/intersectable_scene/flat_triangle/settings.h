#pragma once

#include "lib/settings.h"

namespace intersectable_scene {
namespace flat_triangle {
template <Setting AccelSettings> struct Settings {
  AccelSettings accel_settings;

  template <class Archive> void serialize(Archive &archive) {
    archive(CEREAL_NVP(accel_settings));
  }

  constexpr inline bool operator==(const Settings &) const = default;
};
} // namespace flat_triangle
} // namespace intersectable_scene
