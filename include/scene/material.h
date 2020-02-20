#pragma once

#include "lib/cuda/utils.h"

#include <tiny_obj_loader.h>

#include <Eigen/Core>

namespace scene {
// TODO:
class Material {
public:
  HOST_DEVICE Material() = default;

  Material(const tinyobj::material_t &material) {}

  // outgoing_dir vs incoming_dir irrelevent because of reversability
  HOST_DEVICE float brdf(const Eigen::Vector3f &outgoing_dir,
                         const Eigen::Vector3f &incoming_dir) const {}

  HOST_DEVICE inline float emmited() const {}
  
  HOST_DEVICE inline bool is_mirror() const {}
};
} // namespace scene
