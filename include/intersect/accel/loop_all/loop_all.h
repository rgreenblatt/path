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
  unsigned size;
  AABB aabb;

  constexpr inline const AABB &bounds() const { return aabb; }

  template <Object O> struct IntersectableRef {
    Span<const O> objects;
    const Ref &ref;

    using InfoType = std::tuple<unsigned, typename O::InfoType>;

    HOST_DEVICE inline IntersectionOp<InfoType> intersect(const Ray &ray) const;

    constexpr inline const AABB &bounds() const { return ref.bounds(); }
  };

  template <Object O>
  constexpr inline IntersectableRef<O>
  get_intersectable(Span<const O> objects) const {
    return {objects, *this};
  }
};
} // namespace detail

template <ExecutionModel execution_model> struct LoopAll {
  template <typename B>
  detail::Ref gen(const Settings &, SpanSized<const B> objects,
                  const AABB &aabb) {
    return detail::Ref{static_cast<unsigned>(objects.size()), aabb};
  }
};

template <ExecutionModel exec>
struct LoopAllIsAccel : BoolWrapper<BoundsOnlyAccel<LoopAll<exec>, Settings>> {
};

static_assert(PredicateForAllValues<ExecutionModel>::value<LoopAllIsAccel>);
} // namespace loop_all
} // namespace accel
} // namespace intersect
