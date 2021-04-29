#pragma once

namespace intersect {
namespace accel {
namespace enum_accel {
// consider renaming to BoundOnly something...
enum class AccelType {
  LoopAll,
  NaivePartitionBVH,
  DirTree,
};
} // namespace enum_accel
} // namespace accel
} // namespace intersect
