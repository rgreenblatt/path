#include "intersect/accel/loop_all/loop_all.h"

namespace intersect {
namespace accel {
namespace loop_all {
template <ExecutionModel execution_model>
template <typename B>
detail::Ref LoopAll<execution_model>::gen(const Settings &, SpanSized<const B>,
                                          const AABB &aabb) {
  return detail::Ref{aabb};
}

template struct LoopAll<ExecutionModel::CPU>;
template struct LoopAll<ExecutionModel::GPU>;
} // namespace loop_all
} // namespace accel
} // namespace intersect
