#pragma once

namespace intersect {
namespace accel {
namespace enum_accel {
// consider renaming to BoundOnly something...
enum class AccelType {
  LoopAll,
  KDTree,
  DirTree,
};
} // namespace enum_accel
} // namespace accel
} // namespace intersect
