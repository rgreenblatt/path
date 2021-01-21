#pragma once

#include "intersect/accel/accel.h"
#include "intersect/accel/dir_tree/settings.h"
#include "intersect/accel/s_a_heuristic_settings.h"
#include "lib/attribute.h"
#include "meta/all_values/impl/enum.h"
#include "meta/all_values/predicate_for_all_values.h"

namespace intersect {
namespace accel {
namespace dir_tree {
namespace detail {
// In this case, the Ref type doesn't depend on the ExecutionModel (at least
// not yet...)
struct Ref {
  template <IntersectableAtIdx F>
  ATTR_PURE_NDEBUG HOST_DEVICE inline AccelRet<F>
  intersect_objects(const intersect::Ray &, const F &) const {
    unreachable_unchecked();
  }
};
} // namespace detail

template <ExecutionModel execution_model> class DirTree {
public:
  using Ref = detail::Ref;

  template <Bounded B>
  detail::Ref gen(const Settings &, SpanSized<const B>, const AABB &) {
    // TODO
    unreachable();
    return {};
  }
};

// I think dir tree might be triangle only in practice...
template <ExecutionModel exec>
struct IsAccel : std::bool_constant<BoundsOnlyAccel<DirTree<exec>, Settings>> {
};

static_assert(PredicateForAllValues<ExecutionModel>::value<IsAccel>);
} // namespace dir_tree
} // namespace accel
} // namespace intersect
