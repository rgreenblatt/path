#pragma once

#include "intersect/triangle.h"
#include "intersect/triangle_impl.h"
#include "lib/cuda/utils.h"
#include "lib/unit_vector.h"

#include <Eigen/Core>

namespace scene {
class TriangleData {
public:
  HOST_DEVICE TriangleData() {}

  HOST_DEVICE TriangleData(std::array<UnitVector, 3> normals,
                           unsigned material_idx)
      : normals_(normals), material_idx_(material_idx) {}

  HOST_DEVICE UnitVector get_normal(const Eigen::Vector3f &point,
                                    const intersect::Triangle &triangle) const {
    auto [w0, w1, w2] = triangle.interpolation_values(point);

    return UnitVector::new_normalize(*normals_[0] * w0 + *normals_[1] * w1 +
                                     *normals_[2] * w2);
  }

  HOST_DEVICE const std::array<UnitVector, 3> &normals() const {
    return normals_;
  }

  HOST_DEVICE unsigned material_idx() const { return material_idx_; }

private:
  std::array<UnitVector, 3> normals_;
  unsigned material_idx_;
};
} // namespace scene
