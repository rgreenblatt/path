#pragma once

#include "intersect/triangle.h"
#include "intersect/triangle_impl.h"
#include "lib/attribute.h"
#include "lib/cuda/utils.h"
#include "lib/unit_vector.h"

namespace scene {
struct TriangleData {
  std::array<UnitVector, 3> normals_;
  unsigned material_idx_;

  ATTR_PURE_NDEBUG HOST_DEVICE UnitVector get_normal(
      const Eigen::Vector3f &point, const intersect::Triangle &triangle) const {
    auto [w0, w1, w2] = triangle.interpolation_values(point);

    return UnitVector::new_normalize(*normals_[0] * w0 + *normals_[1] * w1 +
                                     *normals_[2] * w2);
  }

  ATTR_PURE HOST_DEVICE const std::array<UnitVector, 3> &normals() const {
    return normals_;
  }

  ATTR_PURE_NDEBUG HOST_DEVICE TriangleData
  transform(const Eigen::Affine3f &transform) const {
    const auto linear = transform.linear();
    std::array<UnitVector, 3> new_normals{
        UnitVector::new_normalize(linear * *normals_[0]),
        UnitVector::new_normalize(linear * *normals_[1]),
        UnitVector::new_normalize(linear * *normals_[2]),
    };

    return {
        .normals_ = new_normals,
        .material_idx_ = material_idx_,
    };
  }

  ATTR_PURE HOST_DEVICE unsigned material_idx() const { return material_idx_; }
};
} // namespace scene
