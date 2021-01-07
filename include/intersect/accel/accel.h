#pragma once

#include "execution_model/execution_model.h"
#include "intersect/object.h"
#include "lib/settings.h"
#include "lib/span.h"
#include "meta/specialization_of.h"

namespace intersect {
namespace accel {
template <typename T> struct IdxHolder {
  unsigned idx;
  T value;
};

template <typename T>
concept IntersectableAtIdx = requires(const T &ref, unsigned idx,
                                      const Ray &ray) {
  { *ref(idx, ray) }
  ->SpecializationOf<Intersection>;
};

struct MockInfoType : MockCopyable {};

struct MockIntersectableAtIdx : MockNoRequirements {
  IntersectionOp<MockInfoType> operator()(unsigned, const Ray &ray) const;
};

static_assert(IntersectableAtIdx<MockIntersectableAtIdx>);

template <IntersectableAtIdx F>
using AccelRet = IntersectionOp<IdxHolder<
    std::decay_t<decltype(std::declval<F>()(unsigned(), Ray{})->info)>>>;

template <typename V>
concept AccelRef =
    requires(const V &accel_ref, const Ray &ray,
             const MockIntersectableAtIdx &intersectable_at_idx) {
  requires std::copyable<V>;

  { accel_ref.intersect_objects(ray, intersectable_at_idx) }
  ->DecaysTo<AccelRet<MockIntersectableAtIdx>>;
};

namespace detail {
// Settings type is the same for each object, so we don't use an associated
// type
template <typename T, typename Settings, typename B>
concept GeneralAccel = requires(T &accel, const Settings &settings,
                                SpanSized<const B> objects, const AABB &aabb) {
  requires Bounded<B>;
  requires std::default_initializable<T>;
  requires std::movable<T>;
  requires Setting<Settings>;
  typename T::Ref;
  requires AccelRef<typename T::Ref>;

  // generation
  { accel.gen(settings, objects, aabb) }
  ->std::same_as<typename T::Ref>;
};
} // namespace detail

// Accel which only uses bounds and which works on any objects/bounds on input
// We test the concepts on Mocks, but they should work for anything
// Note that BoundsOnlyAccel implies ObjectSpecificAccel
template <typename T, typename Settings>
concept BoundsOnlyAccel = detail::GeneralAccel<T, Settings, MockBounded>;

template <typename T, typename Settings, typename O>
concept ObjectSpecificAccel = detail::GeneralAccel<T, Settings, O>;
} // namespace accel
} // namespace intersect
