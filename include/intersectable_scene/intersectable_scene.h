#pragma once

#include "intersect/object.h"
#include "intersect/ray.h"
#include "material/material.h"

#include <concepts>

namespace intersect {
template <typename T>
concept IntersectableScene = requires(const T &t, const Ray &ray) {
  requires Object<T>;

  requires requires(const Intersection<typename T::InfoType> &intersection) {
    { t.get_normal(intersection) }
    ->std::convertible_to<Eigen::Vector3f>;
    { t.get_material(intersection) }
    ->std::convertible_to<const material::Material &>;
  };
};
} // namespace intersect
