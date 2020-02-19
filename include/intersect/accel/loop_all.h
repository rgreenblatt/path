#pragma once

#include "intersect/intersection.h"
#include "intersect/ray.h"
#include "lib/cuda/utils.h"
#include "lib/execution_model_vector_type.h"
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

    using PrevInfoType = typename std::decay_t<decltype(
        *std::declval<Object>()(Ray()))>::InfoType;
    using NewInfoType = AppendIndexInfoType<PrevInfoType>;
    using IntersectionOpT = IntersectionOp<NewInfoType>;

    HOST_DEVICE IntersectionOpT operator()(const Ray &ray) const {
      IntersectionOpT best_intersection;
      for (unsigned idx = 0; idx < objects_.size(); idx++) {
        auto intersection = objects_[idx](ray);
        best_intersection = optional_min(
            best_intersection, append_index(intersection, idx + offset_));
      }

      return best_intersection;
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
