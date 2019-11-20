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
};

struct PointLight {
  Eigen::Vector3f position;
  Eigen::Array3f attenuation_function;

  PointLight(const Eigen::Vector3f &position,
             const Eigen::Array3f &attenuation_function)
      : position(position), attenuation_function(attenuation_function) {}
  PointLight() {}
};

class Light {
public:
  Color color;
  bool is_directional;

  Light(const Color &color, const DirectionalLight &directional_light)
      : color(color), directional_light(directional_light) {}

  Light(const Color &color, const PointLight &point_light)
      : color(color), point_light(point_light) {}

  template <typename F>
  HOST_DEVICE
  auto visit(const F &f) const {
    if (is_directional) {
      return f(directional_light);
    } else {
      return f(point_light);
    }
  }

private:
  DirectionalLight directional_light;
  PointLight point_light;
};
} // namespace scene
