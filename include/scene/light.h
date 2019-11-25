#pragma once

#include "lib/cuda_utils.h"
#include "scene/material.h"

#include <Eigen/Core>

#include <variant>

namespace scene {
struct DirectionalLight {
  Eigen::Vector3f direction;

  DirectionalLight(const Eigen::Vector3f &direction) : direction(direction) {}

  DirectionalLight() {}

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct PointLight {
  Eigen::Vector3f position;
  Eigen::Array3f attenuation_function;

  PointLight(const Eigen::Vector3f &position,
             const Eigen::Array3f &attenuation_function)
      : position(position), attenuation_function(attenuation_function) {}
  PointLight() {}

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class Light {
public:
  Color color;
  bool is_directional;

  Light(const Color &color, const DirectionalLight &directional_light)
      : color(color), is_directional(true),
        directional_light_(directional_light) {}

  Light(const Color &color, const PointLight &point_light)
      : color(color), is_directional(false), point_light_(point_light) {}

  template <typename F> HOST_DEVICE auto visit(const F &f) const {
    if (is_directional) {
      return f(directional_light_);
    } else {
      return f(point_light_);
    }
  }

private:
  DirectionalLight directional_light_;
  PointLight point_light_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
} // namespace scene
