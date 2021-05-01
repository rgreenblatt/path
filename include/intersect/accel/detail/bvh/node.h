#pragma once

#include "intersect/accel/aabb.h"
#include "lib/start_end.h"
#include "lib/tagged_union.h"
#include "meta/all_values/impl/enum.h"

namespace intersect {
namespace accel {
namespace detail {
namespace bvh {
struct Split {
  unsigned left_idx;
  unsigned right_idx;
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
} // namespace bvh
} // namespace detail
} // namespace accel
} // namespace intersect
