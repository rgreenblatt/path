#pragma once

#include "intersectable_scene/intersectable_scene.h"
#include "lib/settings.h"
#include "scene/scene.h"

// TODO: eventually scene should allow for non triangle
// scenes and alternate material
namespace intersectable_scene {
// this would also have to change with non-tri scene...
template <Intersector I, SceneRefForInfoType<typename I::InfoType> S>
struct SceneGenerated {
  Span<const typename I::InfoType> orig_triangle_idx_to_info;
  IntersectableScene<I, S> intersectable_scene;
};

template <typename T, typename Settings>
concept SceneGenerator = requires(T &gen, const Settings &settings,
                                  const scene::Scene &scene) {
  requires std::default_initializable<T>;
  requires std::movable<T>;
  requires Setting<Settings>;

  typename T::Intersector;
  typename T::SceneRef;

  {
    gen.gen(settings, scene)
    } -> std::same_as<
        SceneGenerated<typename T::Intersector, typename T::SceneRef>>;
};
} // namespace intersectable_scene
