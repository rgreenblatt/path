#pragma once

#include "lib/cuda/utils.h"
#include "lib/span.h"
#include "ray/execution_model.h"
#include "ray/detail/accel/aabb.h"
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

  // TODO union
  unsigned start;
  unsigned end;
  // for x and y divisions, only bound_min is used
  float bound_min;
  float bound_max;

  Type type;

  HOST_DEVICE DirTreeNode(unsigned start, unsigned end)
      : start(start), end(end), type(Type::Indexes) {}

  HOST_DEVICE DirTreeNode(float bound) : bound_min(bound), type(Type::Split) {}

  // z case
  HOST_DEVICE DirTreeNode(float bound_min, float bound_max)
      : bound_min(bound_min), bound_max(bound_max), type(Type::Split) {}

  HOST_DEVICE DirTreeNode() {}
};

// eventually this should be triangle or something...
using BoundingPoints = std::array<Eigen::Vector3f, 8>;

inline BoundingPoints get_bounding(const Eigen::Affine3f &transform_v) {
  auto trans = [&](const Eigen::Vector3f &point) {
    return transform_v * point;
  };

  return {
      trans({0.5f, 0.5f, 0.5f}),   trans({-0.5f, 0.5f, 0.5f}),
      trans({0.5f, -0.5f, 0.5f}),  trans({0.5f, 0.5f, -0.5f}),
      trans({-0.5f, -0.5f, 0.5f}), trans({0.5f, -0.5f, -0.5f}),
      trans({-0.5f, 0.5f, -0.5f}), trans({-0.5f, -0.5f, -0.5f}),
  };
}

struct ALIGN_STRUCT(32) DirTree {
  // is it worth special casing affine? (union or whatever...)
  Eigen::Projective3f transform;
  Span<const DirTreeNode> nodes;
  Span<const Action> actions;

  HOST_DEVICE DirTree(Eigen::Projective3f transform,
                      Span<const DirTreeNode> nodes, Span<const Action> actions)
      : transform(transform), nodes(nodes), actions(actions) {}

  HOST_DEVICE DirTree() {}

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct IdxAABB {
  unsigned idx;
  AABB aabb;

  HOST_DEVICE IdxAABB(unsigned idx, const AABB &aabb) : idx(idx), aabb(aabb) {}

  HOST_DEVICE IdxAABB() {}

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct EdgeIdxAABB {
  IdxAABB idx_aabb;
  bool is_min;

  HOST_DEVICE EdgeIdxAABB(const IdxAABB &idx_aabb, bool is_min)
      : idx_aabb(idx_aabb), is_min(is_min) {}

  HOST_DEVICE EdgeIdxAABB() {}
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

template <ExecutionModel execution_model>
void compute_aabbs(Span<Eigen::Projective3f> transforms,
                   unsigned num_transforms, Span<IdxAABB> aabbs,
                   Span<const BoundingPoints> bounds, unsigned num_bounds);
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
