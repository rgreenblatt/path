#pragma once

#include "intersect/accel/accel.h"
#include "intersect/accel/dir_tree/settings.h"
#include "intersect/accel/s_a_heuristic_settings.h"
#include "intersect/triangle.h"
#include "meta/predicate_for_all_values.h"

namespace intersect {
namespace accel {
namespace dir_tree {
namespace detail {
// In this case, the Ref type doesn't depend on the ExecutionModel (at least
// not yet...)
class Ref {
public:
  HOST_DEVICE Ref() {}

  Ref(const AABB &aabb) : aabb_(aabb) {}

  constexpr inline const AABB &bounds() const { return aabb_; }

  template <Object O> struct IntersectableRef {
    Span<const O> objects;
    const Ref &ref;

    using InfoType = std::tuple<unsigned, typename O::InfoType>;

    constexpr inline IntersectionOp<InfoType> intersect(const Ray &) const {
      // TODO: write and move to impl
      assert(false);

      return thrust::nullopt;
    }

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

template <ExecutionModel execution_model> class DirTree {
public:
  template <Bounded B>
  detail::Ref gen(const Settings &settings, SpanSized<const B> objects,
          const AABB &aabb);
};

// I think dir tree might be triangle only in practice...
template <ExecutionModel exec>
struct DirTreeIsAccel : BoolWrapper<BoundsOnlyAccel<DirTree<exec>, Settings>> {
};

static_assert(PredicateForAllValues<ExecutionModel>::value<DirTreeIsAccel>);
} // namespace dir_tree
} // namespace accel
} // namespace intersect
