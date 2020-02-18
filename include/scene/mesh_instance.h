#pragma once

#include <Eigen/Geometry>

namespace scene {
struct MeshInstance {
  unsigned idx;
  Eigen::Affine3f transform;
};
} // namespace scene
