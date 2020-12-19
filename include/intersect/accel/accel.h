#pragma once

#include "execution_model/execution_model.h"
#include "intersect/accel/kdtree/settings.h"
#include "intersect/accel/s_a_heuristic_settings.h"
#include "intersect/object.h"
#include "lib/span.h"
#include "lib/settings.h"

namespace intersect {
namespace accel {
template <typename V, typename O>
concept AccelIntersectableRef =
    Object<V> &&std::same_as<typename V::InfoType,
                             std::tuple<unsigned, typename O::InfoType>>;

template <typename V, typename O>
concept AccelRef = requires(const V &accel_ref, Span<const O> objects) {
  requires Object<O>;
  requires std::copyable<V>;

  { accel_ref.get_intersectable(objects) } ->
  AccelIntersectableRef<O>;
};

namespace detail {
// Settings type is the same for each object, so we don't use an associated type
template <typename T, typename Settings, typename O, typename B>
concept GeneralAccel = requires(T &accel, const Settings &settings,
                                SpanSized<const O> objects, const AABB &aabb) {
  requires Bounded<B>;
  requires std::default_initializable<T>;
  requires std::movable<T>;
  requires Setting<Settings>;

  // generation
  { accel.gen(settings, objects, aabb) }
  ->AccelRef<O>;
};
} // namespace detail

// Accel which only uses bounds and which works on any objects/bounds on input
// We test the concepts on Mocks, but they should work for anything
// Note that BoundsOnlyAccel implies ObjectSpecificAccel
template <typename T, typename Settings>
concept BoundsOnlyAccel =
    detail::GeneralAccel<T, Settings, MockObject, MockBounded>;


template <typename T, typename Settings, typename O>
concept ObjectSpecificAccel = detail::GeneralAccel<T, Settings, O, O>;
} // namespace accel
} // namespace intersect
