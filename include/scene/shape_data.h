#pragma once

#include "scene/material.h"

#include <Eigen/Dense>

namespace scene {
using Transform = Eigen::Matrix4f;
struct ShapeData {
  Transform transform;
  Material material;

  ShapeData(const Transform &transform, const Material &material)
      : transform(transform), material(material) {}
};
} // namespace scene
