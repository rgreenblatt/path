#pragma once

#include "lib/cuda/utils.h"
#include "lib/eigen_utils.h"
#include "lib/optional.h"

#include <Eigen/Geometry>

#include <iostream>

namespace intersect {
namespace accel {
inline std::tuple<Eigen::Vector3f, Eigen::Vector3f>
get_transformed_bounds(const Eigen::Affine3f &transform,
                       const Eigen::Vector3f &min_bound,
                       const Eigen::Vector3f &max_bound) {
  auto min_transformed_bound = max_eigen_vec();
  auto max_transformed_bound = min_eigen_vec();
  for (auto x_is_min : {false, true}) {
    for (auto y_is_min : {false, true}) {
      for (auto z_is_min : {false, true}) {
        auto get_axis = [&](bool is_min, uint8_t axis) {
          return is_min ? min_bound[axis] : max_bound[axis];
        };
        Eigen::Vector3f transformed_edge =
            transform * Eigen::Vector3f(get_axis(x_is_min, 0),
                                        get_axis(y_is_min, 1),
                                        get_axis(z_is_min, 2));
        min_transformed_bound =
            min_transformed_bound.cwiseMin(transformed_edge);
        max_transformed_bound =
            max_transformed_bound.cwiseMax(transformed_edge);
      }
    }
  }

  return {min_transformed_bound, max_transformed_bound};
}
struct AABB {
  Eigen::Vector3f min_bound;
  Eigen::Vector3f max_bound;

  // implementing bounded
  HOST_DEVICE inline const AABB &bounds() const { return *this; }

  HOST_DEVICE inline AABB transform(const Eigen::Affine3f &transform) const {
    auto [min, max] = get_transformed_bounds(transform, min_bound, max_bound);

    return {min, max};
  }

  HOST_DEVICE inline AABB union_other(const AABB &other) const {
    return {min_bound.cwiseMin(other.min_bound),
            max_bound.cwiseMax(other.max_bound)};
  }

  HOST_DEVICE float surface_area() const {
    auto dims = (max_bound - min_bound).eval();
    return 2 *
           (dims.x() * dims.y() + dims.z() * dims.y() + dims.z() * dims.x());
  }

  // needs to be inline
  HOST_DEVICE Optional<float>
  solveBoundingIntersection(const Eigen::Vector3f &point,
                            const Eigen::Vector3f &inv_direction) const {
    auto t_0 = (min_bound - point).cwiseProduct(inv_direction).eval();
    auto t_1 = (max_bound - point).cwiseProduct(inv_direction).eval();
    auto t_min = t_0.cwiseMin(t_1);
    auto t_max = t_0.cwiseMax(t_1);

    float max_of_min = t_min.maxCoeff();
    float min_of_max = t_max.minCoeff();

    if (max_of_min <= min_of_max) {
      return max_of_min;
    } else {
      return nullopt_value;
    }
  }

  friend std::ostream &operator<<(std::ostream &s, const AABB &v) {
    s << "min bound: "
      << "\n"
      << v.min_bound << "\n"
      << "max bound: "
      << "\n"
      << v.max_bound << "\n";

    return s;
  }
};
} // namespace accel
} // namespace intersect
