#pragma once

#include "ray/cube.h"
#include <thrust/optional.h>

namespace ray {
namespace detail {
struct KDTreeNode;

struct KDTreeSplit {
  unsigned left_index;
  unsigned right_index;
  float division_point;
  KDTreeSplit(unsigned left_index, unsigned right_index, float division_point)
      : left_index(left_index), right_index(right_index),
        division_point(division_point) {}
  KDTreeSplit() {}
};

struct KDTreeNode {
  Eigen::Vector3f min_bound;
  Eigen::Vector3f max_bound;

  KDTreeNode() {}

  KDTreeNode(const KDTreeSplit &split, const Eigen::Vector3f &min_bound,
             const Eigen::Vector3f &max_bound)
      : KDTreeNode(min_bound, max_bound) {
    split_ = split;
    is_split_ = true;
  }

  KDTreeNode(const std::array<unsigned, 2> &data,
             const Eigen::Vector3f &min_bound, const Eigen::Vector3f &max_bound)
      : KDTreeNode(min_bound, max_bound) {
    data_ = data;
    is_split_ = false;
  }

  HOST_DEVICE thrust::optional<float>
  solveBoundingIntersectionOld(const Eigen::Vector3f &point,
                               const Eigen::Vector3f &direction) const {
    Eigen::Vector3f kd_space_point = point.cwiseProduct(scale) + translate;

    float half_plus_epsilon = 0.5f + epsilon;
    if (std::abs(kd_space_point.x()) < half_plus_epsilon &&
        std::abs(kd_space_point.y()) < half_plus_epsilon &&
        std::abs(kd_space_point.z()) < half_plus_epsilon) {
      // point is inside bounding box
      return 0.0f;
    }

    return solve_cube<false>(kd_space_point, direction.cwiseProduct(scale),
                             false);
  }

  // needs to be inline
  HOST_DEVICE thrust::optional<float>
  solveBoundingIntersection(const Eigen::Vector3f &point,
                            const Eigen::Vector3f &inv_direction) const {
    auto t_0 = (min_bound - point).cwiseProduct(inv_direction).eval();
    auto t_1 = (max_bound - point).cwiseProduct(inv_direction).eval();
    auto t_min = t_0.cwiseMin(t_1);
    auto t_max = t_0.cwiseMax(t_1);

    float max_of_min = t_min.maxCoeff();
    float min_of_max = t_max.minCoeff();

    return make_optional(max_of_min <= min_of_max, max_of_min);
  }

  template <typename FSplit, typename FData>
  constexpr auto case_split_or_data(const FSplit &split_case,
                                    const FData &data_case) const {
    if (is_split_) {
      return split_case(split_);
    } else {
      return data_case(data_);
    }
  }

private:
  KDTreeNode(const Eigen::Vector3f &min_bound, const Eigen::Vector3f &max_bound)
      : min_bound(min_bound), max_bound(max_bound),
        scale(1.0f / (max_bound - min_bound).array()),
        translate(
            (-min_bound - (max_bound - min_bound) * 0.5f).cwiseProduct(scale)) {
  }

  Eigen::Vector3f scale;
  Eigen::Vector3f translate;
  bool is_split_;
  KDTreeSplit split_;
  std::array<unsigned, 2> data_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct Bounds {
  Eigen::Vector3f min;
  Eigen::Vector3f center;
  Eigen::Vector3f max;

  Bounds() {}
  Bounds(const Eigen::Vector3f &min, const Eigen::Vector3f &center,
         const Eigen::Vector3f &max)
      : min(min), center(center), max(max) {}
};

std::vector<KDTreeNode> construct_kd_tree(scene::ShapeData *shapes,
                                          unsigned num_shapes);
} // namespace detail
} // namespace ray
