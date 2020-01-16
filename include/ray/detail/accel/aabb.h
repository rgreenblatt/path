#pragma once

#include "lib/cuda/utils.h"
#include "ray/detail/projection.h"

#include <Eigen/Geometry>
#include <thrust/optional.h>

namespace ray {
namespace detail {
namespace accel {
class AABB {
public:
  HOST_DEVICE
  AABB() {}

  HOST_DEVICE
  AABB(const Eigen::Vector3f &min_bound, const Eigen::Vector3f &max_bound)
      : min_bound_(min_bound), max_bound_(max_bound) {}

  HOST_DEVICE const Eigen::Vector3f &get_min_bound() const {
    return min_bound_;
  }

  HOST_DEVICE const Eigen::Vector3f &get_max_bound() const {
    return max_bound_;
  }

  // needs to be inline
  HOST_DEVICE thrust::optional<float>
  solveBoundingIntersection(const Eigen::Vector3f &point,
                            const Eigen::Vector3f &inv_direction) const {
    auto t_0 = (min_bound_ - point).cwiseProduct(inv_direction).eval();
    auto t_1 = (max_bound_ - point).cwiseProduct(inv_direction).eval();
    auto t_min = t_0.cwiseMin(t_1);
    auto t_max = t_0.cwiseMax(t_1);

    float max_of_min = t_min.maxCoeff();
    float min_of_max = t_max.minCoeff();

    if (max_of_min <= min_of_max) {
      return max_of_min;
    } else {
      return thrust::nullopt;
    }
  }

private:
  Eigen::Vector3f min_bound_;
  Eigen::Vector3f max_bound_;
};

inline std::tuple<Eigen::Vector3f, Eigen::Vector3f> get_transformed_bounds(
    const Eigen::Projective3f &transform,
    const Eigen::Vector3f &min_bound = Eigen::Vector3f::Ones() * -0.5f,
    const Eigen::Vector3f &max_bound = Eigen::Vector3f::Ones() * 0.5f) {
  Eigen::Vector3f min_transformed_bound(std::numeric_limits<float>::max(),
                                        std::numeric_limits<float>::max(),
                                        std::numeric_limits<float>::max());
  Eigen::Vector3f max_transformed_bound(std::numeric_limits<float>::lowest(),
                                        std::numeric_limits<float>::lowest(),
                                        std::numeric_limits<float>::lowest());
  for (auto x_is_min : {false, true}) {
    for (auto y_is_min : {false, true}) {
      for (auto z_is_min : {false, true}) {
        auto get_axis = [&](bool is_min, uint8_t axis) {
          return is_min ? min_bound[axis] : max_bound[axis];
        };
        Eigen::Vector3f transformed_edge = apply_projective_point(
            Eigen::Vector3f(get_axis(x_is_min, 0), get_axis(y_is_min, 1),
                            get_axis(z_is_min, 2)),
            transform);
        min_transformed_bound =
            min_transformed_bound.cwiseMin(transformed_edge);
        max_transformed_bound =
            max_transformed_bound.cwiseMax(transformed_edge);
      }
    }
  }

  return std::make_tuple(min_transformed_bound, max_transformed_bound);
}
} // namespace accel
} // namespace detail
} // namespace ray
