#pragma once

#include "lib/settings.h"

namespace intersectable_scene {
namespace flat_triangle {
template <Setting AccelSettings> struct Settings {
  AccelSettings accel_settings;

  SETTING_BODY(Settings, accel_settings);
};
} // namespace flat_triangle
} // namespace intersectable_scene
