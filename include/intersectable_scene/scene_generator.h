#pragma once

#include "intersectable_scene/intersectable_scene.h"
#include "lib/settings.h"
#include "scene/scene.h"

namespace intersectable_scene {
// TODO: eventually scene should allow for non triangle
// scenes and alternate material
template <typename T, typename Settings>
concept SceneGenerator = requires(T &gen, const Settings &settings,
                                  const scene::Scene &scene) {
  requires std::default_initializable<T>;
  requires std::movable<T>;
  requires Setting<Settings>;

  { gen.gen(settings, scene) }
  ->IntersectableSceneForBSDF<scene::Material::BSDFT>;
};
} // namespace intersectable_scene