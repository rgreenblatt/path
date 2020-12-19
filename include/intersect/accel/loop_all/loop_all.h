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
class Ref {
public:
  // TODO: why is this constructor needed...
  HOST_DEVICE Ref() {}

  Ref(const AABB &aabb) : aabb_(aabb) {}

  constexpr inline const AABB &bounds() const { return aabb_; }

  template <Object O> struct IntersectableRef {
    Span<const O> objects;
    const Ref &ref;

    using InfoType = std::tuple<unsigned, typename O::InfoType>;

    constexpr inline IntersectionOp<InfoType> intersect(const Ray &ray) const;

    constexpr inline const AABB &bounds() const { return ref.bounds(); }
  };

  template <Object O>
  constexpr inline IntersectableRef<O>
  get_intersectable(Span<const O> objects) const {
    return {objects, *this};
  }

private:
  AABB aabb_;
};
} // namespace detail

template <ExecutionModel execution_model> struct LoopAll {
  template <typename B>
  detail::Ref gen(const Settings &settings, SpanSized<const B> objects,
          const AABB &aabb);
};

template <ExecutionModel exec>
struct LoopAllIsAccel : BoolWrapper<BoundsOnlyAccel<LoopAll<exec>, Settings>> {
};

static_assert(PredicateForAllValues<ExecutionModel>::value<LoopAllIsAccel>);
} // namespace loop_all
} // namespace accel
} // namespace intersect
