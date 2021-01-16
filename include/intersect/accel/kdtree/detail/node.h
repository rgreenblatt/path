#pragma once

#include "intersect/accel/aabb.h"
#include "lib/start_end.h"
#include "lib/tagged_union.h"
#include "meta/all_values_enum.h"

namespace intersect {
namespace accel {
namespace kdtree {
namespace detail {
struct Bounds {
  AABB aabb;
  Eigen::Vector3f center;
};

struct Split {
  unsigned left_index;
  unsigned right_index;
  float division_point;
};

enum class NodeType {
  Split,
  Items,
};

using NodeValue = TaggedUnion<NodeType, Split, StartEnd<unsigned>>;

struct Node {
  NodeValue value;
  AABB aabb;
};
} // namespace detail
} // namespace kdtree
} // namespace accel
} // namespace intersect
