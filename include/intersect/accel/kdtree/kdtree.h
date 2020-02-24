#pragma once

#include "intersect/accel/aabb.h"
#include "lib/cuda/utils.h"
#include "lib/span.h"

#include <Eigen/Geometry>
#include <thrust/optional.h>

namespace intersect {
namespace accel {
namespace kdtree {

std::vector<KDTreeNode<AABB>> construct_kd_tree(Span<unsigned> indexes,

                                                unsigned target_depth,
                                                unsigned target_shapes_per);
} // namespace kdtree
} // namespace accel
} // namespace intersect
