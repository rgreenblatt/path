#pragma once

#include "render/settings.h"

namespace generate_data {
namespace full_scene {
static void amend_config(render::Settings &settings) {
  // TODO: is sobel fine???
  settings.rendering_equation_settings.back_cull_emission = false;
}
} // namespace full_scene
} // namespace generate_data
