#pragma once

#include "scene/material.h"
#include "lib/cuda_utils.h"

#include <Eigen/Dense>
#include <Eigen/Core>

namespace scene {
using Transform = Eigen::Affine3f;
class ShapeData {
public:
  HOST_DEVICE const Transform &get_world_to_object() const {
    return world_to_object_;
  }

  HOST_DEVICE const Eigen::Matrix3f &
  get_object_normal_to_world() const {
    return object_normal_to_world_;
  }

  HOST_DEVICE void set_transform(const Transform &transform) {
    transform_ = transform;
    world_to_object_ = transform_.inverse();
    object_normal_to_world_ =
        transform_.linear().transpose().inverse();
  }

  HOST_DEVICE ShapeData(const Transform &transform,
                                const Material &material)
      : material_(material) {
    set_transform(transform);
  }

private:
  Transform transform_;
  Material material_;
  Transform world_to_object_;
  Eigen::Matrix3f object_normal_to_world_;
};
} // namespace scene
