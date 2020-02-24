#pragma once

#include "execution_model/execution_model_vector_type.h"
#include "intersect/accel/accel.h"
#include "lib/cuda/utils.h"

#include <thrust/copy.h>

namespace intersect {
namespace accel {
template <ExecutionModel execution_model, Object O>
struct AccelImpl<AccelType::LoopAll, execution_model, O> {
  AccelImpl() {}

  class Ref {
  public:
    HOST_DEVICE Ref() {}

    Ref(SpanSized<const O> objects, unsigned offset, const AABB &aabb)
        : objects_(objects), offset_(offset), aabb_(aabb) {}

    HOST_DEVICE inline const O &get(unsigned idx) const {
      return objects_[idx - offset_];
    }

    constexpr static AccelType inst_type = AccelType::LoopAll;
    constexpr static ExecutionModel inst_execution_model = execution_model;
    using InstO = O;

  private:
    SpanSized<const O> objects_;
    unsigned offset_;

    AABB aabb_;

    friend struct IntersectableImpl<Ref>;
  };

  Ref gen(const AccelSettings<AccelType::LoopAll> &, Span<const O> objects,
          unsigned start, unsigned end, const AABB &aabb) {
    if constexpr (execution_model == ExecutionModel::GPU) {
      unsigned size = end - start;
      store_.resize(size);
      thrust::copy(objects.begin() + start, objects.begin() + end,
                   store_.begin());

      return {store_, start, aabb};
    } else {
      return {objects.slice(start, end), start, aabb};
    }
  }

private:
  struct NoneType {};

  std::conditional_t<execution_model == ExecutionModel::GPU,
                     ExecVector<execution_model, O>, NoneType>
      store_;
};

template <typename V>
concept LoopAllRef = AccelRefOfType<V, AccelType::LoopAll>;
} // namespace accel

// TODO: consider moving to impl file
template <accel::LoopAllRef Ref> struct IntersectableImpl<Ref> {
  using Temp = float;
  static HOST_DEVICE inline auto intersect(const Ray &ray, const Ref &ref) {
    using O = typename Ref::InstO;
    using IntersectionO = IntersectableT<O>;
    using PrevInfoType = typename IntersectionO::Intersection;
    using NewInfoType = AppendIndexInfoType<PrevInfoType>;
    using IntersectionOpT = IntersectionOp<NewInfoType>;

    IntersectionOpT best_intersection;

    for (unsigned idx = 0; idx < ref.objects_.size(); idx++) {
      auto intersection =
          IntersectableImpl<O>::intersect(ray, ref.objects_[idx]);
      best_intersection = optional_min(
          best_intersection, append_index(intersection, idx + ref.offset_));
    }

    return best_intersection;
  }
};

template <accel::LoopAllRef Ref> struct BoundedImpl<Ref> {
  static HOST_DEVICE inline const accel::AABB &bounds(const Ref &ref) {
    return ref.aabb();
  }
};
} // namespace intersect
