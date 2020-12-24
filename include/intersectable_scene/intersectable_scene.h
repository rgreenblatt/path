#pragma once

#include "bsdf/material.h"
#include "intersect/object.h"
#include "intersect/ray.h"
#include "meta/decays_to.h"
#include "meta/specialization_of.h"

#include <concepts>

namespace intersectable_scene {
template <typename T>
concept IntersectableScene = requires(const T &t, const intersect::Ray &ray) {
  requires intersect::Object<T>;
  typename T::B;
  requires bsdf::BSDF<typename T::B>;

  requires requires(
      const intersect::Intersection<typename T::InfoType> &intersection) {
    { t.get_normal(intersection, ray) }
    ->std::convertible_to<Eigen::Vector3f>;
    { t.get_material(intersection) }
    ->DecaysTo<bsdf::Material<typename T::B>>;
  };
};

template <typename T, typename B> concept IntersectableSceneForBSDF = requires {
  requires IntersectableScene<T>;
  requires std::same_as<B, typename T::S>;
};
} // namespace intersectable_scene
