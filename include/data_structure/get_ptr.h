#pragma once

#include "lib/concepts.h"
#include "lib/trait.h"

#include <thrust/device_vector.h>

#include <array>
#include <concepts>
#include <type_traits>
#include <utility>
#include <vector>

template <template <typename> class Impl, typename T, typename Elem>
concept GetPtrTrait = requires(T &&t) {
  typename Impl<T>;
  { Impl<T>::get(std::forward<T>(t)) }
  ->std::convertible_to<Elem *>;
};

template <template <typename> class Impl, typename T, typename Elem>
struct SatisfiesTrait {
  static_assert(GetPtrTrait<Impl, T, Elem>);

  static constexpr bool value = GetPtrTrait<Impl, T, Elem>;
};

template <typename Meta, typename T> struct GetPtrTraitImpl;

template <IsBaseCaseType V> struct GetPtrTraitImpl<StartT, V> {};

template <typename Base, typename T>
    requires StdArraySpecialization<T> ||
    SpecializationOf<T, std::vector> struct GetPtrTraitImpl<Base, T> : Base {
  static auto get(T &&v) { return v.data(); }
};

template <typename Base, typename T>
requires SpecializationOf<T, thrust::device_vector> struct GetPtrTraitImpl<Base,
                                                                           T>
    : Base {
  static auto get(T &&t) { return thrust::raw_pointer_cast(t.data()); }
};

template <typename T, typename Elem> struct GetPtrTraitS {
  template <template <typename> class Impl, typename TSub>
  using Sat = SatisfiesTrait<Impl, TSub, Elem>;

  using type = get_t<GetPtrTraitImpl, Sat, T>;
};

template <typename T, typename Elem>
using GetPtrT = typename GetPtrTraitS<T, Elem>::type;

template <typename Elem> struct GetPtrElem {
  template <typename T> using type = GetPtrT<T, Elem>;
};

template <typename T, typename Elem> concept GetPtr = requires(T &&t) {
  typename GetPtrT<T, Elem>;
  GetPtrTrait<GetPtrElem<Elem>::template type, T, Elem>;
};
