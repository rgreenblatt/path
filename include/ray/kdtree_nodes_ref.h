#pragma once

#include "ray/best_intersection.h"
#include "ray/kdtree.h"
#include "lib/span.h"
#include <thrust/optional.h>

namespace ray {
namespace detail {
struct KDTreeNodesRef {
  Span<KDTreeNode<AABB>, false> nodes;
  unsigned num_shape;

  KDTreeNodesRef(KDTreeNode<AABB> *nodes, unsigned num_nodes,
                 unsigned num_shape)
      : nodes(nodes, num_nodes), num_shape(num_shape) {}

  KDTreeNodesRef() {}
};
} // namespace detail
} // namespace ray
