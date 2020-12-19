#pragma once

#include "intersect/object.h"
#include "intersect/ray.h"
#include "material/material.h"

#include <concepts>

namespace intersectable_scene {
template <typename T>
concept IntersectableScene = requires(const T &t, const intersect::Ray &ray) {
  requires intersect::Object<T>;

  requires requires(
      const intersect::Intersection<typename T::InfoType> &intersection) {
    { t.get_normal(intersection, ray) }
    ->std::convertible_to<Eigen::Vector3f>;
    { t.get_material(intersection) }
    ->std::same_as<const material::Material&>;
  };
};
} // namespace intersectable_scene
