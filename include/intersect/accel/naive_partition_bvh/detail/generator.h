#pragma once

#include "execution_model/execution_model_vector_type.h"
#include "intersect/accel/aabb.h"
#include "intersect/accel/naive_partition_bvh/naive_partition_bvh.h"
#include "intersect/accel/naive_partition_bvh/settings.h"
#include "lib/span.h"

#include <tuple>

namespace intersect {
namespace accel {
using namespace detail::bvh;
namespace naive_partition_bvh {
// should only be used from detail context
using namespace detail;

template <ExecutionModel exec> class NaivePartitionBVH<exec>::Generator {
public:
  Generator() = default;

  RefPerm<BVH<>> gen(const Settings &settings, SpanSized<Bounds> objects);

private:
  unsigned partition(SpanSized<Bounds> bounds, SpanSized<unsigned> idxs,
                     uint8_t axis);
  void kth_smallest(SpanSized<Bounds> bounds, SpanSized<unsigned> idxs,
                    size_t k, uint8_t axis);

  AABB get_bounding(SpanSized<Bounds> bounds);

  NodeTup create_node(SpanSized<Bounds> bounds, SpanSized<unsigned> idxs,
                      NodeGroup<HostVector> &nodes, unsigned start_idx,
                      unsigned depth);

  template <typename T> using ExecVecT = ExecVector<exec, T>;

  NodeGroup<ExecVecT> nodes_out_;

  Settings settings_;
};
} // namespace naive_partition_bvh
} // namespace accel
} // namespace intersect
