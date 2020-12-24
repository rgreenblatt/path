#pragma once

#include "execution_model/execution_model_vector_type.h"
#include "intersect/accel/accel.h"
#include "intersect/accel/loop_all/settings.h"
#include "meta/predicate_for_all_values.h"

namespace intersect {
namespace accel {
namespace loop_all {
namespace detail {
// In this case, the Ref type doesn't depend on the ExecutionModel
struct Ref {
  AABB aabb;
  unsigned size;

  constexpr const AABB &bounds() const { return aabb; }

  template <Object O>
  HOST_DEVICE inline AccelRet<O> intersect_objects(const intersect::Ray &ray,
                                                   Span<const O> objects) const;
};
} // namespace detail

template <ExecutionModel execution_model> struct LoopAll {
  template <typename B>
  detail::Ref gen(const Settings &, SpanSized<const B> objects,
                  const AABB &aabb) {
    return detail::Ref{aabb, static_cast<unsigned>(objects.size())};
  }
};

template <ExecutionModel exec>
struct IsAccel : BoolWrapper<BoundsOnlyAccel<LoopAll<exec>, Settings>> {};

static_assert(PredicateForAllValues<ExecutionModel>::value<IsAccel>);
} // namespace loop_all
} // namespace accel
} // namespace intersect
