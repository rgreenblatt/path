#pragma once

#include "meta/all_values/all_values.h"
#include "meta/all_values/impl/tuple.h"
#include "meta/tuple.h"

#include <boost/hana/ext/std/integer_sequence.hpp>
#include <boost/hana/unpack.hpp>

#include <type_traits>
#include <utility>

// apply predicate to all values and return true if it holds for all of them
template <AllValuesEnumerable... T> struct PredicateForAllValues {
  static constexpr auto values = AllValues<MetaTuple<T...>>;

  // unfortunately, this nesting is needed due to the dependence on T...
  // Pred should always be convertible to bool (std::bool_constant)
  template <template <T...> class Pred>
  static constexpr bool value =
  // TODO: gcc work around (https://gcc.gnu.org/bugzilla/show_bug.cgi?id=86883)
#ifdef __clang__
      boost::hana::unpack(
          std::make_index_sequence<values.size()>{}, [](auto... i) {
            return (... && [&](auto idx) {
              return boost::hana::unpack(
                  std::make_index_sequence<sizeof...(T)>{},
                  [&](auto... j) -> bool {
                    return Pred<meta_tuple_at<j>(values[idx])...>{};
                  });
            }(i));
          });
#else
      true;
#endif
};
