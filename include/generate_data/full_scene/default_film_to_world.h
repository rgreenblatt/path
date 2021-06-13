#pragma once

#include "scene/camera.h"

namespace generate_data {
namespace full_scene {
Eigen::Affine3f default_film_to_world() {
  return scene::get_camera_transform(
      UnitVector::new_normalize({0.f, 0.f, -1.f}),
      UnitVector::new_normalize({0.f, 1.f, 0.f}), {0.f, 0.f, 10.f}, 45.f, 1.f);
}
} // namespace full_scene
} // namespace generate_data
