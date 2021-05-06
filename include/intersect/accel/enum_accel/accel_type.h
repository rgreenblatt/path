#pragma once

namespace intersect {
namespace accel {
namespace enum_accel {
// consider renaming to TriangleOnly something...
// (or making it so all are bounds only...)
enum class AccelType {
  LoopAll,
  NaivePartitionBVH,
  SBVH,
  DirectionGrid,
};
} // namespace enum_accel
} // namespace accel
} // namespace intersect
