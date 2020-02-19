#pragma once

#include "intersect/intersection.h"
#include "intersect/ray.h"
#include "lib/cuda/utils.h"

#include <Eigen/Core>
#include <thrust/optional.h>

namespace intersect {
class Triangle {
public:
  HOST_DEVICE Triangle() {}

  HOST_DEVICE Triangle(std::array<Eigen::Vector3f, 3> vertices)
      : vertices_(vertices) {}

  HOST_DEVICE inline IntersectionOp<std::tuple<>>
  operator()(const Ray &ray) const;

private:
  std::array<Eigen::Vector3f, 3> vertices_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
} // namespace intersect
