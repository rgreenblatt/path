#pragma once

#include "intersect/accel/accel.h"
#include "intersect/accel/s_a_heuristic_settings.h"

namespace intersect {
namespace accel {
template <ExecutionModel execution_model, Object O>
struct AccelImpl<AccelType::DirTree, execution_model, O> {
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

    constexpr static AccelType inst_type = AccelType::DirTree;
    constexpr static ExecutionModel inst_execution_model = execution_model;
    using InstO = O;

  private:
    AABB aabb_;
  };

  Ref gen(const AccelSettings<AccelType::DirTree> &, Span<const O>, unsigned,
          unsigned, const AABB &aabb) {
    assert(false);
    return Ref(aabb);
  }
};

template <typename V>
concept DirTreeRef = AccelRefOfType<V, AccelType::DirTree>;
} // namespace accel

// TODO: consider moving to impl file
template <accel::DirTreeRef Ref> struct IntersectableImpl<Ref> {
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

template <accel::DirTreeRef Ref> struct BoundedImpl<Ref> {
  static HOST_DEVICE inline const accel::AABB &bounds(const Ref &ref) {
    return ref.aabb();
  }
};
} // namespace intersect
