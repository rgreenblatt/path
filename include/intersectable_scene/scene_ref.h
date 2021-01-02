#pragma once

#include "bsdf/material.h"
#include "intersect/intersection.h"
#include "intersect/ray.h"
#include "meta/decays_to.h"

#include <concepts>

namespace intersectable_scene {
template <typename T, typename InfoType>
concept SceneRef =
    requires(const T &t, const intersect::Ray &ray,
             const intersect::Intersection<InfoType> &intersection) {
  typename T::B;
  requires bsdf::BSDF<typename T::B>;

  { t.get_normal(intersection, ray) }
  ->std::convertible_to<UnitVector>;
  { t.get_material(intersection) }
  ->DecaysTo<bsdf::Material<typename T::B>>;
};
} // namespace intersectable_scene
