#pragma once

#include "intersect/intersection.h"
#include "intersect/ray.h"
#include "lib/cuda/utils.h"
#include "lib/execution_model.h"
#include "lib/span.h"

#include <Eigen/Core>
#include <thrust/copy.h>
#include <thrust/optional.h>

namespace intersect {
namespace accel {
template <ExecutionModel execution_model, typename Object> class LoopAll {
private:
  class LoopAllRef {
  public:
    LoopAllRef() {}

    HOST_DEVICE inline auto operator()(const Ray &ray) const;

    HOST_DEVICE inline const Object &get(unsigned idx) const {
      return objects_[idx - offset_];
    }

  private:
    LoopAllRef(SpanSized<const Object> objects, unsigned offset)
        : objects_(objects), offset_(offset) {}

    SpanSized<const Object> objects_;
    unsigned offset_;

    friend class LoopAll;
  };

public:
  LoopAll() {}

  using RefType = LoopAllRef;

  RefType gen(Span<const Object> objects, unsigned start, unsigned end) {
    if constexpr (execution_model == ExecutionModel::GPU) {
      unsigned size = end - start;
      store_.resize(size);
      thrust::copy(objects.begin() + start, objects.begin() + end,
                   store_.begin());

      return {store_, start};
    } else {
      return {objects.slice(start, end), start};
    }
  }

private:
  struct NoneType {};

  std::conditional_t<execution_model == ExecutionModel::GPU,
                     ExecVector<execution_model, Object>, NoneType>
      store_;
};
} // namespace accel
} // namespace intersect
