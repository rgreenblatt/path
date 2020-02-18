#pragma once

#include "lib/cuda/utils.h"
#include "intersect/ray.h"

#include <Eigen/Core>
#include <thrust/optional.h>

namespace intersect {
class Triangle {
public:
  HOST_DEVICE Triangle() {}

  HOST_DEVICE Triangle(std::array<Eigen::Vector3f, 3> vertices)
      : vertices_(vertices) {}

  HOST_DEVICE inline thrust::optional<float>
  get_intersection(const Ray &ray) const;

private:
  std::array<Eigen::Vector3f, 3> vertices_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
} // namespace detail
