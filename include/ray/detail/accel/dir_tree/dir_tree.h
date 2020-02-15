#pragma once

#include "lib/cuda/utils.h"
#include "lib/execution_model.h"
#include "lib/span.h"
#include "ray/detail/accel/dir_tree/sphere_partition.h"

#include <Eigen/Geometry>

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
class ALIGN_STRUCT(16) DirTreeNode {
public:
  HOST_DEVICE DirTreeNode(unsigned start, unsigned end)
      : start_end_({start, end}), type_(Type::StartEnd) {}

  HOST_DEVICE DirTreeNode(float split_point, unsigned left, unsigned right)
      : split_({split_point, left, right}), type_(Type::Split) {}

  HOST_DEVICE DirTreeNode() {}

  template <typename F> auto visit(F &f) const {
    switch (type_) {
    case Type::Split:
      return f(split_);
    case Type::StartEnd:
      return f(start_end_);
    }
  }

  struct Split {
    float split_point;
    unsigned left;
    unsigned right;
  };

  struct StartEnd {
    unsigned start;
    unsigned end;
  };

private:
  enum class Type {
    Split,
    StartEnd,
  };

  union {
    Split split_;
    StartEnd start_end_;
  };

  Type type_;
};

// eventually this should be triangle or something...
class ALIGN_STRUCT(32) DirTree {
public:
  // is it worth special casing affine? (union or whatever...)
  //   - only affine now I think...
  // is it worth special casing rotation?
  //   - only rotation now I think...

  HOST_DEVICE DirTree(Eigen::Projective3f transform, unsigned idx)
      : transform_(transform), idx_(idx) {}

  HOST_DEVICE unsigned start_node_idx() const { return idx_ + 1; }

  HOST_DEVICE unsigned min_max_idx() const { return idx_; }

  HOST_DEVICE const Eigen::Projective3f &transform() const {
    return transform_;
  }

  HOST_DEVICE DirTree() = default;

private:
  Eigen::Projective3f transform_;
  unsigned idx_;
};

class ALIGN_STRUCT(32) DirTreeLookup {
public:
  HOST_DEVICE DirTreeLookup() = default;

  DirTreeLookup(Span<const DirTree> dir_trees,
                const HalfSpherePartition &partition,
                Span<const DirTreeNode> nodes,
                Span<const float> min_sorted_values,
                Span<const float> min_sorted_inclusive_maxes,
                Span<const unsigned> min_sorted_indexes,
                Span<const float> max_sorted_values,
                Span<const float> max_sorted_inclusive_mins,
                Span<const unsigned> max_sorted_indexes,
                Span<const float> x_min, Span<const float> y_min,
                Span<const float> z_min, Span<const float> x_max,
                Span<const float> y_max, Span<const float> z_max)
      : dir_trees_(dir_trees), partition_(partition), nodes_(nodes),
        min_sorted_values_(min_sorted_values),
        min_sorted_inclusive_maxes_(min_sorted_inclusive_maxes),
        min_sorted_indexes_(min_sorted_indexes),
        max_sorted_values_(max_sorted_values),
        max_sorted_inclusive_mins_(max_sorted_inclusive_mins),
        max_sorted_indexes_(max_sorted_indexes), x_min_(x_min), y_min_(y_min),
        z_min_(z_min), x_max_(x_max), y_max_(y_max), z_max_(z_max) {}

  inline HOST_DEVICE std::tuple<const DirTree &, bool>
  getDirTree(const Eigen::Vector3f &direction) const {
    auto [idx, is_flipped] = partition_.get_closest(direction);
    return {dir_trees_[idx], is_flipped};
  }

  inline HOST_DEVICE Span<const DirTreeNode> nodes() const { return nodes_; }

  inline HOST_DEVICE Span<const float> min_sorted_values() const {
    return min_sorted_values_;
  }

  inline HOST_DEVICE Span<const float> min_sorted_inclusive_maxes() const {
    return min_sorted_inclusive_maxes_;
  }

  inline HOST_DEVICE Span<const unsigned> min_sorted_indexes() const {
    return min_sorted_indexes_;
  }

  inline HOST_DEVICE Span<const float> max_sorted_values() const {
    return max_sorted_values_;
  }

  inline HOST_DEVICE Span<const float> max_sorted_inclusive_mins() const {
    return max_sorted_inclusive_mins_;
  }

  inline HOST_DEVICE Span<const unsigned> max_sorted_indexes() const {
    return max_sorted_indexes_;
  }

  inline HOST_DEVICE Span<const float> x_min() const { return x_min_; }

  inline HOST_DEVICE Span<const float> y_min() const { return y_min_; }

  inline HOST_DEVICE Span<const float> z_min() const { return z_min_; }

  inline HOST_DEVICE Span<const float> x_max() const { return x_max_; }

  inline HOST_DEVICE Span<const float> y_max() const { return y_max_; }

  inline HOST_DEVICE Span<const float> z_max() const { return z_max_; }

private:
  Span<const DirTree> dir_trees_;
  HalfSpherePartition partition_;

  Span<const DirTreeNode> nodes_;

  Span<const float> min_sorted_values_;
  Span<const float> min_sorted_inclusive_maxes_;
  Span<const unsigned> min_sorted_indexes_;

  Span<const float> max_sorted_values_;
  Span<const float> max_sorted_inclusive_mins_;
  Span<const unsigned> max_sorted_indexes_;

  Span<const float> x_min_;
  Span<const float> y_min_;
  Span<const float> z_min_;

  Span<const float> x_max_;
  Span<const float> y_max_;
  Span<const float> z_max_;
};
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
