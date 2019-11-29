#pragma once

#include "ray/best_intersection.h"
#include "ray/kdtree.h"
#include <thrust/optional.h>

namespace ray {
namespace detail {
struct ByTypeDataGPU {
  thrust::optional<BestIntersectionNormalUV> *intersections;
  KDTreeNode *nodes;
  unsigned root_node_count;
  scene::Shape shape_type;
  unsigned start_shape;
  unsigned num_shape;

  ByTypeDataGPU(
      thrust::optional<detail::BestIntersectionNormalUV> *intersections,
      detail::KDTreeNode *nodes, unsigned root_node_count,
      scene::Shape shape_type, unsigned start_shape, unsigned num_shape)
      : intersections(intersections), nodes(nodes),
        root_node_count(root_node_count), shape_type(shape_type),
        start_shape(start_shape), num_shape(num_shape) {}

  ByTypeDataGPU() {}
};
} // namespace detail
} // namespace ray
