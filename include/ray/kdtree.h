#pragma once

#include "lib/cuda_utils.h"
#include "scene/shape_data.h"

#include <thrust/optional.h>
#include <Eigen/Geometry>

namespace ray {
namespace detail {
struct KDTreeSplit {
  unsigned left_index;
  unsigned right_index;
  float division_point;
  KDTreeSplit(unsigned left_index, unsigned right_index, float division_point)
      : left_index(left_index), right_index(right_index),
        division_point(division_point) {}
  HOST_DEVICE
  KDTreeSplit() {}
};

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

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template <typename Contents> struct KDTreeNode {
  HOST_DEVICE KDTreeNode() {}

  HOST_DEVICE KDTreeNode(const KDTreeSplit &split, const Contents &contents)
      : contents_(contents), is_split_(true), split_(split) {
    split_ = split;
    is_split_ = true;
  }

  KDTreeNode(const std::array<unsigned, 2> &data, const Contents &contents)
      : contents_(contents), is_split_(false), data_(data) {}

  template <typename FSplit, typename FData>
  constexpr auto case_split_or_data(const FSplit &split_case,
                                    const FData &data_case) const {
    if (is_split_) {
      return split_case(split_);
    } else {
      return data_case(data_);
    }
  }

  HOST_DEVICE bool get_is_split() const { return is_split_; }

  template <typename F> auto transform(const F &f) const {
    auto get_out = [&](const auto &v) {
      return KDTreeNode<decltype(f(contents_))>(v, f(contents_));
    };

    if (is_split_) {
      return get_out(split_);
    } else {
      return get_out(data_);
    }
  }

  HOST_DEVICE const Contents &get_contents() const { return contents_; }

private:
  Contents contents_;

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

inline std::tuple<Eigen::Vector3f, Eigen::Vector3f>
get_shape_bounds(const scene::ShapeData &shape) {
  Eigen::Vector3f min_bound(std::numeric_limits<float>::max(),
                            std::numeric_limits<float>::max(),
                            std::numeric_limits<float>::max());
  Eigen::Vector3f max_bound(std::numeric_limits<float>::lowest(),
                            std::numeric_limits<float>::lowest(),
                            std::numeric_limits<float>::lowest());
  for (auto x : {-0.5f, 0.5f}) {
    for (auto y : {-0.5f, 0.5f}) {
      for (auto z : {-0.5f, 0.5f}) {
        Eigen::Vector3f transformed_edge =
            shape.get_transform() * Eigen::Vector3f(x, y, z);
        min_bound = min_bound.cwiseMin(transformed_edge);
        max_bound = max_bound.cwiseMax(transformed_edge);
      }
    }
  }

  return std::make_tuple(min_bound, max_bound);
}

std::vector<KDTreeNode<AABB>> construct_kd_tree(scene::ShapeData *shapes,
                                                unsigned num_shapes,
                                                unsigned target_depth,
                                                unsigned target_shapes_per);
} // namespace detail
} // namespace ray
