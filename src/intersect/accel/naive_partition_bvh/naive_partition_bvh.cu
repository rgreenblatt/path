#include "intersect/accel/naive_partition_bvh/detail/generator.h"
#include "intersect/accel/naive_partition_bvh/naive_partition_bvh.h"

namespace intersect {
namespace accel {
namespace naive_partition_bvh {
template <ExecutionModel exec> NaivePartitionBVH<exec>::NaivePartitionBVH() {
  gen_ = std::make_unique<Generator>();
}

template <ExecutionModel exec>
NaivePartitionBVH<exec>::~NaivePartitionBVH() = default;

template <ExecutionModel exec>
NaivePartitionBVH<exec>::NaivePartitionBVH(NaivePartitionBVH &&) = default;

template <ExecutionModel exec>
NaivePartitionBVH<exec> &
NaivePartitionBVH<exec>::operator=(NaivePartitionBVH &&) = default;

template <ExecutionModel exec>
RefPerm<BVH> NaivePartitionBVH<exec>::gen_internal(const Settings &settings) {
  return gen_->gen(settings, bounds_);
}

template class NaivePartitionBVH<ExecutionModel::CPU>;
#ifndef CPU_ONLY
template class NaivePartitionBVH<ExecutionModel::GPU>;
#endif
} // namespace naive_partition_bvh
} // namespace accel
} // namespace intersect
