#include "intersect/accel/sbvh/detail/generator.h"
#include "intersect/accel/sbvh/sbvh.h"

namespace intersect {
namespace accel {
namespace sbvh {
template <ExecutionModel exec> SBVH<exec>::SBVH() {
  gen_ = std::make_unique<Generator>();
}

template <ExecutionModel exec> SBVH<exec>::~SBVH() = default;

template <ExecutionModel exec> SBVH<exec>::SBVH(SBVH &&) = default;

template <ExecutionModel exec>
SBVH<exec> &SBVH<exec>::operator=(SBVH &&) = default;

template <ExecutionModel exec>
RefPerm<BVH> SBVH<exec>::gen(const Settings &settings,
                             SpanSized<const Triangle> triangles) {
  return gen_->gen(settings, triangles);
}

template class SBVH<ExecutionModel::CPU>;
#ifndef CPU_ONLY
template class SBVH<ExecutionModel::GPU>;
#endif
} // namespace sbvh
} // namespace accel
} // namespace intersect
