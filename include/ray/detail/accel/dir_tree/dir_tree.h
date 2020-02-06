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
struct Action {
  unsigned shape_idx;
  float min_dist;
  float max_dist;

  HOST_DEVICE Action(unsigned shape_idx, float min_dist, float max_dist)
      : shape_idx(shape_idx), min_dist(min_dist), max_dist(max_dist) {}

  HOST_DEVICE
  Action() {}
};

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
  // TODO
  Span<const Action> actions;

  HOST_DEVICE DirTree(Eigen::Projective3f transform,
                      Span<const DirTreeNode> nodes, Span<const Action> actions)
      : transform(transform), nodes(nodes), actions(actions) {}

  HOST_DEVICE DirTree() {}
};

class ALIGN_STRUCT(32) DirTreeLookup {
public:
  DirTreeLookup() {}

  DirTreeLookup(Span<const DirTree> dir_trees, unsigned camera_idx,
                unsigned start_lights_idx, unsigned start_partition_idx,
                const HalfSpherePartition &partition)
      : dir_trees_(dir_trees), camera_idx_(camera_idx),
        start_lights_idx_(start_lights_idx),
        start_partition_idx_(start_partition_idx), partition_(partition) {}

  inline HOST_DEVICE const DirTree &getCameraTraversalData() const {
    return dir_trees_[camera_idx_];
  }

  inline HOST_DEVICE const DirTree &
  getLightTraversalData(unsigned light_idx) const {
    return dir_trees_[start_lights_idx_ + light_idx];
  }

  inline HOST_DEVICE const DirTree &
  getGeneralTraversalData(const Eigen::Vector3f &direction) const {
    return dir_trees_[start_partition_idx_ + partition_.get_closest(direction)];
  }

private:
  Span<const DirTree> dir_trees_;
  unsigned camera_idx_;
  unsigned start_lights_idx_;
  unsigned start_partition_idx_;
  HalfSpherePartition partition_;
};
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
