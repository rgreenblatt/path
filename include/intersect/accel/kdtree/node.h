#pragma once

#include "intersect/accel/aabb.h"
#include "lib/attribute.h"
#include "lib/cuda/utils.h"

#include <Eigen/Geometry>

namespace intersect {
namespace accel {
namespace kdtree {
namespace detail {
struct Bounds {
  AABB aabb;
  Eigen::Vector3f center;
};

struct KDTreeSplit {
  unsigned left_index;
  unsigned right_index;
  float division_point;
};

template <typename Contents> struct KDTreeNode {
  HOST_DEVICE KDTreeNode() {}

  HOST_DEVICE KDTreeNode(const KDTreeSplit &split, const Contents &contents)
      : contents_(contents), is_split_(true), split_(split) {
    split_ = split;
    is_split_ = true;
  }

  HOST_DEVICE KDTreeNode(const std::array<unsigned, 2> &data,
                         const Contents &contents)
      : contents_(contents), is_split_(false), data_(data) {}

  template <typename FSplit, typename FData>
  HOST_DEVICE auto case_split_or_data(const FSplit &split_case,
                                      const FData &data_case) const {
    if (is_split_) {
      return split_case(split_);
    } else {
      return data_case(data_);
    }
  }

  ATTR_PURE HOST_DEVICE bool get_is_split() const { return is_split_; }

  template <typename F> HOST_DEVICE auto transform(const F &f) const {
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
};
} // namespace detail
} // namespace kdtree
} // namespace accel
} // namespace intersect
