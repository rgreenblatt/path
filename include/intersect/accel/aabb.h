#pragma once

#include "lib/attribute.h"
#include "lib/cuda/utils.h"
#include "lib/eigen_utils.h"
#include "lib/optional.h"
#include "lib/unit_vector.h"

#include <Eigen/Geometry>

#include <iostream>

namespace intersect {
namespace accel {

HOST_DEVICE inline Eigen::Vector3f
get_inv_direction(const Eigen::Vector3f &direction) {
  Eigen::Vector3f direction_no_zeros = direction;

  for (unsigned i = 0; i < unsigned(direction_no_zeros.size()); ++i) {
    float &v = direction_no_zeros[i];
    if (v == 0.0f || v == -0.0f) {
      v = 1e-20f;
    }
  }

  return 1.0f / direction_no_zeros.array();
}

struct AABB {
  Eigen::Vector3f min_bound;
  Eigen::Vector3f max_bound;

  ATTR_PURE_NDEBUG HOST_DEVICE static inline AABB empty() {
    return {.min_bound = max_eigen_vec(), .max_bound = min_eigen_vec()};
  }

  // implementing bounded
  ATTR_PURE_NDEBUG HOST_DEVICE inline const AABB &bounds() const {
    return *this;
  }

  ATTR_PURE_NDEBUG HOST_DEVICE inline AABB
  transform(const Eigen::Affine3f &transform) const {
    AABB out = AABB::empty();
    for (auto x_is_min : {false, true}) {
      for (auto y_is_min : {false, true}) {
        for (auto z_is_min : {false, true}) {
          auto get_axis = [&](bool is_min, uint8_t axis) {
            return is_min ? min_bound[axis] : max_bound[axis];
          };
          Eigen::Vector3f transformed_point =
              transform * Eigen::Vector3f(get_axis(x_is_min, 0),
                                          get_axis(y_is_min, 1),
                                          get_axis(z_is_min, 2));
          out = out.union_point(transformed_point);
        }
      }
    }

    return out;
  }

  ATTR_PURE_NDEBUG HOST_DEVICE inline AABB
  union_other(const AABB &other) const {
    return {min_bound.cwiseMin(other.min_bound),
            max_bound.cwiseMax(other.max_bound)};
  }

  ATTR_PURE_NDEBUG HOST_DEVICE inline AABB
  union_point(const Eigen::Vector3f &point) const {
    return {min_bound.cwiseMin(point), max_bound.cwiseMax(point)};
  }

  ATTR_PURE_NDEBUG HOST_DEVICE bool
  contains(const Eigen::Vector3f &point) const {
    return (min_bound.array() <= point.array()).all() &&
           (max_bound.array() >= point.array()).all();
  }

  ATTR_PURE_NDEBUG HOST_DEVICE float surface_area() const {
    auto dims = (max_bound - min_bound).eval();

    // handle "empty" case
    if ((dims.array() < 0.f).any()) {
      return 0.;
    }

    return 2 *
           (dims.x() * dims.y() + dims.z() * dims.y() + dims.z() * dims.x());
  }

  ATTR_PURE_NDEBUG HOST_DEVICE Eigen::Vector3f centroid() const {
    return (min_bound + max_bound) / 2.f;
  }

  // contains both intersection points
  struct BoundingIntersection {
    float t_min;
    float t_max;
  };

  // needs to be inline
  ATTR_PURE_NDEBUG HOST_DEVICE inline std::optional<BoundingIntersection>
  solve_bounding_intersection(const Eigen::Vector3f &point,
                              const Eigen::Vector3f &inv_direction) const {
    auto t_0 = (min_bound - point).cwiseProduct(inv_direction).eval();
    auto t_1 = (max_bound - point).cwiseProduct(inv_direction).eval();
    auto all_t_min = t_0.cwiseMin(t_1);
    auto all_t_max = t_0.cwiseMax(t_1);

    float overall_t_min = all_t_min.maxCoeff();
    float overall_t_max = all_t_max.minCoeff();

    if (overall_t_min <= overall_t_max) {
      return BoundingIntersection{
          .t_min = overall_t_min,
          .t_max = overall_t_max,
      };
    } else {
      return std::nullopt;
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
