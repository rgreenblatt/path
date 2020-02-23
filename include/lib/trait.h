#pragma once

#include "lib/concepts.h"

#include <vector>

template <typename T> struct BaseCaseType { using type = T; };

template <typename T> concept IsBaseCaseType = requires {
  std::same_as<T, BaseCaseType<typename T::type>>;
};

// guarantees:
// if implemention, then must satisfy trait concepty
// if instantion, then there is implemention
struct StartT {};

template <template <typename> class Impl,
          template <template <typename> class, typename> class SatisfiesTrait,
          typename T>
concept TraitSatisfiable = requires(T t) {
  typename SatisfiesTrait<Impl, T>;
}
&&SatisfiesTrait<Impl, T>::value;

template <template <typename, typename> class Impl,
          template <template <typename> class, typename> class SatisfiesTrait,
          typename T>
struct CheckTrait {
  template <typename TSub>
  using WrappedImpl = Impl<Impl<StartT, BaseCaseType<TSub>>, TSub>;

  static_assert(TraitSatisfiable<WrappedImpl, SatisfiesTrait, T>);
  using type = WrappedImpl<T>;
};

template <template <typename, typename> class Impl,
          template <template <typename> class, typename> class SatisfiesTrait,
          typename T>
using get_t = typename CheckTrait<Impl, SatisfiesTrait, T>::type;
