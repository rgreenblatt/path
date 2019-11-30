#pragma once

#include "lib/cuda_utils.h"
#include "scene/material.h"
#include "ray/ray_utils.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace scene {
using Transform = Eigen::Affine3f;
using UVPosition = Eigen::Array2f;

class ALIGN_STRUCT(32) ShapeData {
public:
  HOST_DEVICE const Transform &get_transform() const {
    return transform_;
  }

  HOST_DEVICE const Transform &get_world_to_object() const {
    return world_to_object_;
  }
  
  HOST_DEVICE const Material &get_material() const { return material_; }

  HOST_DEVICE const Eigen::Matrix3f &get_object_normal_to_world() const {
    return object_normal_to_world_;
  }

  HOST_DEVICE void set_transform(const Transform &transform) {
    transform_ = transform;
    world_to_object_ = transform_.inverse();
    object_normal_to_world_ = transform_.linear().transpose().inverse();
  }

  HOST_DEVICE ShapeData(const Transform &transform, const Material &material)
      : material_(material) {
    set_transform(transform);
  }
  
  HOST_DEVICE ShapeData() {}

private:
  Transform transform_;
  Transform world_to_object_;
  Eigen::Matrix3f object_normal_to_world_;
  Material material_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
} // namespace scene
