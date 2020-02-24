#pragma once

#include "intersect/impl/triangle_impl.h"
#include "intersect/triangle.h"
#include "lib/cuda/utils.h"

#include <Eigen/Core>

namespace scene {
class TriangleData {
public:
  HOST_DEVICE TriangleData() {}

  HOST_DEVICE TriangleData(std::array<Eigen::Vector3f, 3> normals,
                           unsigned material_idx)
      : normals_(normals), material_idx_(material_idx) {}

  HOST_DEVICE Eigen::Vector3f
  get_normal(const Eigen::Vector3f &point,
             const intersect::Triangle &triangle) const {
    return triangle.interpolate_values(point, normals_);
  }

  HOST_DEVICE const std::array<Eigen::Vector3f, 3> &normals() const {
    return normals_;
  }

  HOST_DEVICE unsigned material_idx() const { return material_idx_; }

private:
  std::array<Eigen::Vector3f, 3> normals_;
  unsigned material_idx_;
};
} // namespace scene
