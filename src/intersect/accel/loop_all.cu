#include "intersect/accel/loop_all.h"
#include "intersect/impl/triangle_impl.h"
#include "intersect/transformed_object.h"

namespace intersect {
namespace accel {
template <ExecutionModel execution_model, Object O>
typename AccelImpl<AccelType::LoopAll, execution_model, O>::Ref
AccelImpl<AccelType::LoopAll, execution_model, O>::gen(
    const AccelSettings<AccelType::LoopAll> &, Span<const O> objects,
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

template struct AccelImpl<AccelType::LoopAll, ExecutionModel::CPU,
                          intersect::Triangle>;
template struct AccelImpl<AccelType::LoopAll, ExecutionModel::CPU,
                          intersect::TransformedObject>;
template struct AccelImpl<AccelType::LoopAll, ExecutionModel::GPU,
                          intersect::Triangle>;
template struct AccelImpl<AccelType::LoopAll, ExecutionModel::GPU,
                          intersect::TransformedObject>;
} // namespace accel
} // namespace intersect
