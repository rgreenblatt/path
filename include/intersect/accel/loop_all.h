#pragma once

#include "execution_model/execution_model_vector_type.h"
#include "intersect/accel/accel.h"
#include "lib/cuda/utils.h"

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
          unsigned start, unsigned end, const AABB &aabb);

private:
  struct NoneType {};

  std::conditional_t<execution_model == ExecutionModel::GPU,
                     ExecVector<execution_model, O>, NoneType>
      store_;
};

template <typename V>
concept LoopAllRef = AccelRefOfType<V, AccelType::LoopAll>;
} // namespace accel
} // namespace intersect
