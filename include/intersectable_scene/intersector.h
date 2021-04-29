#pragma once

#include "intersect/intersectable.h"
#include "intersect/ray.h"
#include "lib/array_vec.h"
#include "lib/span.h"
#include "meta/all_values/impl/enum.h"
#include "meta/all_values/pick_type.h"

namespace intersectable_scene {
template <typename V, typename T>
concept IntersectionsOpSpanFor = std::same_as<
    V,
    Span<const intersect::IntersectionOp<typename std::decay_t<T>::InfoType>>>;

template <typename T>
concept BulkIntersector = requires(const T &t, T &t_mut,
                                   SpanSized<intersect::Ray> ray_span) {
  { t.max_size() } -> std::convertible_to<unsigned>;

  typename std::decay_t<T>::InfoType;
  { t_mut.get_intersections(ray_span) } -> IntersectionsOpSpanFor<T>;
};

template <typename T, typename InfoType>
concept BulkIntersectorForInfoType =
    BulkIntersector<T> && std::same_as<typename T::InfoType, InfoType>;

template <typename T>
concept Intersector = requires(const T &t, T &t_mut, const intersect::Ray &ray,
                               unsigned size) {
  { T::individually_intersectable } -> DecaysTo<bool>;
  typename T::InfoType;

  requires BulkIntersector<T> || T::individually_intersectable;

  requires intersect::IntersectableForInfoType<T, typename T::InfoType> ||
      !T::individually_intersectable;
};

template <typename T, typename InfoType>
concept IntersectorForInfoType =
    Intersector<T> && std::same_as<typename T::InfoType, InfoType>;
} // namespace intersectable_scene
