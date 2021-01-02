#pragma once

#include "intersect/intersectable.h"
#include "intersect/ray.h"
#include "intersectable_scene/ray_writer.h"
#include "intersectable_scene/scene_ref.h"
#include "lib/span.h"

namespace intersectable_scene {
template <typename T>
concept IntersectableScene = requires(const T &t, T &t_mut,
                                      const intersect::Ray &ray,
                                      unsigned size) {
  { T::individually_intersectable }
  ->DecaysTo<bool>;

  requires requires {
    { t.max_size() }
    ->std::convertible_to<unsigned>;

    { t_mut.ray_writer(size) }
    ->RayWriter;

    { t_mut.get_intersections() }
    ->std::same_as<Span<const intersect::IntersectionOp<typename T::InfoType>>>;
  }
  || T::individually_intersectable;

  requires requires {
    { t.intersectable() }
    ->intersect::IntersectableWithInfoType<typename T::InfoType>;
  }
  || !T::individually_intersectable;

  { t.scene() }
  ->SceneRef<typename T::InfoType>;
};

template <typename T, typename B>
concept IntersectableSceneForBSDF = requires(const T &t) {
  requires IntersectableScene<T>;
  requires bsdf::BSDF<B>;
  requires std::same_as<typename decltype(t.scene())::B, B>;
};
} // namespace intersectable_scene
