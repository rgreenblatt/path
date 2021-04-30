#pragma once

#include "execution_model/execution_model_vector_type.h"
#include "intersect/accel/aabb.h"
#include "intersect/accel/naive_partition_bvh/detail/node.h"
#include "intersect/accel/naive_partition_bvh/naive_partition_bvh.h"
#include "intersect/accel/naive_partition_bvh/settings.h"
#include "lib/span.h"

#include <tuple>

namespace intersect {
namespace accel {
namespace naive_partition_bvh {
// should only be used from detail context
using namespace detail;

template <ExecutionModel exec> class NaivePartitionBVH<exec>::Generator {
public:
  Generator() = default;

  RefPerm<Ref> gen(const Settings &settings, SpanSized<Bounds> objects);

private:
  unsigned partition(unsigned start, unsigned end, uint8_t axis);
  void kth_smallest(size_t start, size_t end, size_t k, uint8_t axis);

  AABB get_bounding(unsigned start, unsigned end);

  unsigned construct(unsigned start_shape, unsigned end_shape, unsigned depth);

  bool terminate_here(unsigned start, unsigned end);

  HostVector<unsigned> indexes_;
  Span<Bounds> bounds_;
  HostVector<Node> nodes_;

  template <typename T> using ExecVecT = ExecVector<exec, T>;

  ExecVecT<Node> nodes_out_;

  Settings settings_;
};
} // namespace naive_partition_bvh
} // namespace accel
} // namespace intersect
