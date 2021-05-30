#pragma once

#include "bsdf/material.h"
#include "intersect/intersection.h"
#include "intersect/ray.h"
#include "meta/decays_to.h"

#include <concepts>

namespace intersectable_scene {
template <typename T>
concept SceneRef = requires {
  typename T::B;
  requires bsdf::BSDF<typename T::B>;
  typename T::InfoType;

  requires requires(
      const T &t, const intersect::Ray &ray,
      const intersect::Intersection<typename T::InfoType> &intersection,
      const Eigen::Vector3f &point, const typename T::InfoType &info) {
    { t.get_normal(point, info) } -> std::convertible_to<UnitVector>;
    { t.get_material(info) } -> DecaysTo<bsdf::Material<typename T::B>>;
  };
};

template <typename T, typename InfoType>
concept SceneRefForInfoType =
    SceneRef<T> && std::same_as<typename T::InfoType, InfoType>;

template <typename T, typename InfoType, typename B>
concept SceneRefForInfoTypeBSDF =
    SceneRefForInfoType<T, InfoType> && std::same_as<typename T::B, B>;
} // namespace intersectable_scene
