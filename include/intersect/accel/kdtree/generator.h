#pragma once

#include "execution_model/execution_model_vector_type.h"
#include "intersect/accel/aabb.h"
#include "intersect/accel/kdtree/kdtree.h"
#include "intersect/accel/kdtree/node.h"
#include "intersect/accel/kdtree/settings.h"
#include "lib/span.h"

namespace intersect {
namespace accel {
namespace kdtree {
template <ExecutionModel execution_model>
class KDTree<execution_model>::Generator {
public:
  Generator() = default;

  std::tuple<SpanSized<const detail::KDTreeNode<AABB>>, Span<const unsigned>>
  gen(const Settings &settings, SpanSized<detail::Bounds> objects);

private:
  unsigned partition(unsigned start, unsigned end, uint8_t axis);
  void kth_smallest(size_t start, size_t end, size_t k, uint8_t axis);

  AABB get_bounding(unsigned start, unsigned end);

  unsigned construct(unsigned start_shape, unsigned end_shape, unsigned depth);

  bool terminate_here(unsigned start, unsigned end);

  HostVector<unsigned> indexes_;
  Span<detail::Bounds> bounds_;
  HostVector<detail::KDTreeNode<AABB>> nodes_;

  template <typename T> using ExecVecT = ExecVector<execution_model, T>;

  ExecVecT<unsigned> indexes_out_;
  ExecVecT<detail::KDTreeNode<AABB>> nodes_out_;

  Settings settings_;
};
} // namespace kdtree
} // namespace accel
} // namespace intersect
