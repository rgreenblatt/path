#pragma once

#include "lib/concepts.h"
#include "lib/trait.h"

#include <thrust/device_vector.h>

#include <array>
#include <concepts>
#include <vector>

template <template <typename> class Impl, typename T>
concept GetSizeTrait = requires(T &&t) {
  typename Impl<T>;
  { Impl<T>::get(std::forward<T>(t)) }
  ->std::convertible_to<std::size_t>;
};

template <template <typename> class Impl, typename T>
struct SatisfiesGetSizeTrait {
  static_assert(GetSizeTrait<Impl, T>);

  static constexpr bool value = GetSizeTrait<Impl, T>;
};

template <typename Meta, typename T> struct GetSizeTraitImpl;

template <IsBaseCaseType V> struct GetSizeTraitImpl<StartT, V> {};

template <typename Base, typename T>
    requires StdArraySpecialization<T> || SpecializationOf<T, std::vector> ||
    SpecializationOf<T, thrust::device_vector> struct GetSizeTraitImpl<Base, T>
    : Base {
  static auto get(T &&v) { return v.size(); }
};

template <typename T> struct GetSizeTraitS {
  template <template <typename> class Impl, typename TSub>
  using Sat = SatisfiesGetSizeTrait<Impl, TSub>;

  using type = get_t<GetSizeTraitImpl, Sat, T>;
};

template <typename T> using GetSizeT = typename GetSizeTraitS<T>::type;

template <typename T> concept GetSize = requires(T &&t) {
  typename GetSizeT<T>;
  GetSizeTrait<GetSizeT, T>;
};
