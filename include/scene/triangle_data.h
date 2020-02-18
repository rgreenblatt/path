#pragma once

#include <Eigen/Core>

namespace scene {
class TriangleData {
public:
  TriangleData(std::array<Eigen::Vector3f, 3> normals, unsigned material_idx,
               std::array<Eigen::Vector3f, 3> colors)
      : normals_(normals), material_idx_(material_idx), colors_(colors) {}

  const std::array<Eigen::Vector3f, 3> &normals() const { return normals_; }

  unsigned material_idx() const { return material_idx_; }

  const std::array<Eigen::Vector3f, 3> &colors() const { return colors_; }

private:
  std::array<Eigen::Vector3f, 3> normals_;
  unsigned material_idx_;
  std::array<Eigen::Vector3f, 3> colors_;
};
} // namespace scene
