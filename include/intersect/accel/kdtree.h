#pragma once

#include "intersect/accel/accel.h"

namespace intersect {
namespace accel {
template <ExecutionModel execution_model, Object O>
struct AccelImpl<AccelType::KDTree, execution_model, O> {
  AccelImpl() {}

  class Ref {
  public:
    HOST_DEVICE Ref() {}

    Ref(const AABB &aabb) : aabb_(aabb) {}

    // TODO change to ref
    HOST_DEVICE inline O get(unsigned) const {
      // TODO
      assert(false);
      return O();
      /* return objects_[idx - offset_]; */
    }

    constexpr static AccelType inst_type = AccelType::KDTree;
    constexpr static ExecutionModel inst_execution_model = execution_model;
    using InstO = O;

  private:
    AABB aabb_;
  };

  Ref gen(const AccelSettings<AccelType::KDTree> &, Span<const O>, unsigned,
          unsigned, const AABB &aabb) {
    // TODO
    assert(false);
    return Ref(aabb);
  }
};

template <typename V> concept KDTreeRef = AccelRefOfType<V, AccelType::KDTree>;
} // namespace accel

// TODO: consider moving to impl file
template <accel::KDTreeRef Ref> struct IntersectableImpl<Ref> {
  static HOST_DEVICE inline auto intersect(const Ray &, const Ref &) {
    // TODO
    assert(false);

    using O = typename Ref::InstO;
    using IntersectionO = IntersectableT<O>;
    using PrevInfoType = typename IntersectionO::Intersection;
    using NewInfoType = AppendIndexInfoType<PrevInfoType>;
    using IntersectionOpT = IntersectionOp<NewInfoType>;

    return IntersectionOpT{};
  }
};

template <accel::KDTreeRef Ref> struct BoundedImpl<Ref> {
  static HOST_DEVICE inline const accel::AABB &bounds(const Ref &ref) {
    return ref.aabb();
  }
};
} // namespace intersect
