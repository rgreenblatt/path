#pragma once

#include "execution_model/execution_model_vector_type.h"
#include "execution_model/thrust_data.h"
#include "intersect/accel/accel.h"
#include "intersect/accel/kdtree/node.h"

namespace intersect {
namespace accel {
namespace kdtree {
template <ExecutionModel execution_model> class Generator;
}
template <ExecutionModel execution_model, Object O>
struct AccelImpl<AccelType::KDTree, execution_model, O> {
  class Ref {
  public:
    HOST_DEVICE Ref() {}

    Ref(SpanSized<const kdtree::KDTreeNode<AABB>> nodes, Span<const O> objects,
        unsigned offset, Span<const unsigned> global_idx_to_local_idx,
        Span<const unsigned> local_idx_to_global_idx, const AABB &aabb)
        : nodes_(nodes), objects_(objects), offset_(offset),
          global_idx_to_local_idx_(global_idx_to_local_idx),
          local_idx_to_global_idx_(local_idx_to_global_idx), aabb_(aabb) {}

    HOST_DEVICE inline const O &get(unsigned idx) const {
      return objects_[global_idx_to_local_idx_[idx - offset_]];
    }

    HOST_DEVICE inline const AABB &aabb() const { return aabb_; }

    constexpr static AccelType inst_type = AccelType::KDTree;
    constexpr static ExecutionModel inst_execution_model = execution_model;
    using InstO = O;

  private:
    SpanSized<const kdtree::KDTreeNode<AABB>> nodes_;

    Span<const O> objects_;

    unsigned offset_;
    Span<const unsigned> global_idx_to_local_idx_;
    Span<const unsigned> local_idx_to_global_idx_;

    AABB aabb_;

    friend struct IntersectableImpl<Ref>;
  };

  AccelImpl();

  ~AccelImpl();

  AccelImpl(AccelImpl &&);

  AccelImpl(const AccelImpl &) = delete;

  Ref gen(const AccelSettings<AccelType::KDTree> &, Span<const O>, unsigned,
          unsigned, const AABB &aabb);

private:
  std::unique_ptr<kdtree::Generator<execution_model>> gen_;

  template <typename T> using ExecVecT = ExecVector<execution_model, T>;

  std::vector<kdtree::Bounds> bounds_;

  ExecVecT<unsigned> global_idx_to_local_idx_;
  ExecVecT<O> unordered_objects_;
  ExecVecT<O> ordered_objects_;

  ThrustData<execution_model> thrust_data;
};
template <typename V> concept KDTreeRef = AccelRefOfType<V, AccelType::KDTree>;
} // namespace accel
} // namespace intersect
