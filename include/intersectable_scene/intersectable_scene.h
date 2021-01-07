#pragma once

#include "intersectable_scene/intersector.h"
#include "intersectable_scene/scene_ref.h"

namespace intersectable_scene {
template <Intersector I, SceneRefForInfoType<typename I::InfoType> S>
struct IntersectableScene {
  I intersector;
  S scene;
};
}; // namespace intersectable_scene
