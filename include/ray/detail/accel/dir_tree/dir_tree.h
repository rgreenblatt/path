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
struct ALIGN_STRUCT(16) DirTreeNode {
  enum class Type {
    Indexes,
    Split,
  };

  // TODO: union
  float split_point;
  unsigned start;
  unsigned end;
  unsigned left;
  unsigned right;

  Type type;

  HOST_DEVICE DirTreeNode(unsigned start, unsigned end)
      : start(start), end(end), type(Type::Indexes) {}

  HOST_DEVICE DirTreeNode(float split_point, unsigned left, unsigned right)
      : split_point(split_point), left(left), right(right), type(Type::Split) {}

  HOST_DEVICE DirTreeNode() {}
};

// eventually this should be triangle or something...
struct ALIGN_STRUCT(32) DirTree {
  // is it worth special casing affine? (union or whatever...)
  // is it worth special casing rotation?
  Eigen::Projective3f transform;
  Span<const DirTreeNode> nodes;

  Span<const float> min_sorted_values;
  Span<const float> min_sorted_inclusive_maxes;
  Span<const unsigned> min_sorted_indexes;

  Span<const float> max_sorted_values;
  Span<const float> max_sorted_inclusive_mins;
  Span<const unsigned> max_sorted_indexes;

  // TODO: indexes should become triangles/shapes?

  HOST_DEVICE DirTree(Eigen::Projective3f transform,
                      Span<const DirTreeNode> nodes,
                      Span<const float> min_sorted_values,
                      Span<const float> min_sorted_inclusive_maxes,
                      Span<const unsigned> min_sorted_indexes,
                      Span<const float> max_sorted_values,
                      Span<const float> max_sorted_inclusive_mins,
                      Span<const unsigned> max_sorted_indexes)
      : transform(transform), nodes(nodes),
        min_sorted_values(min_sorted_values),
        min_sorted_inclusive_maxes(min_sorted_inclusive_maxes),
        min_sorted_indexes(min_sorted_indexes),
        max_sorted_values(max_sorted_values),
        max_sorted_inclusive_mins(max_sorted_inclusive_mins),
        max_sorted_indexes(max_sorted_indexes) {}

  HOST_DEVICE DirTree() = default;
};

class ALIGN_STRUCT(32) DirTreeLookup {
public:
  HOST_DEVICE DirTreeLookup() = default;

  DirTreeLookup(Span<const DirTree> dir_trees,
#if 0
                unsigned camera_idx, unsigned start_lights_idx,
                unsigned start_partition_idx,
#endif
                const HalfSpherePartition &partition)
      : dir_trees_(dir_trees),
#if 0
        camera_idx_(camera_idx), start_lights_idx_(start_lights_idx),
        start_partition_idx_(start_partition_idx),
#endif
        partition_(partition) {
  }

#if 0
  inline HOST_DEVICE const DirTree &getCameraDirTree() const {
    return dir_trees_[camera_idx_];
  }

  inline HOST_DEVICE const DirTree &
  getLightDirTree(unsigned light_idx) const {
    return dir_trees_[start_lights_idx_ + light_idx];
  }
#endif

  inline HOST_DEVICE std::tuple<const DirTree &, bool>
  getDirTree(const Eigen::Vector3f &direction) const {
    auto [idx, is_flipped] = partition_.get_closest(direction);
    return {
      dir_trees_[
#if 0
      start_partition_idx_ +
#endif
          idx],
          is_flipped
    };
  }

private:
  Span<const DirTree> dir_trees_;
#if 0
  unsigned camera_idx_;
  unsigned start_lights_idx_;
  unsigned start_partition_idx_;
#endif
  HalfSpherePartition partition_;
};
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
