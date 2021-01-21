#pragma once

#include "meta/as_tuple/as_tuple.h"

#include <boost/hana/ext/std/integer_sequence.hpp>
#include <boost/hana/ext/std/integral_constant.hpp>
#include <boost/hana/unpack.hpp>
#include <cereal/name_value_pair.hpp>

#include <utility>

namespace cereal {

namespace as_tuple {
namespace detail {
template <typename T, typename Archive, typename Tuple>
constexpr void archive(Archive &ar, Tuple &&tup) {
  constexpr auto size = meta_tuple_size_v<AsTupleT<T>>;
  if constexpr (size != 0) {
    boost::hana::unpack(std::make_index_sequence<size>{}, [&](auto... i) {
      ar(make_nvp(std::string(T::as_tuple_strs()[i]), tup[i])...);
    });
  }
}
} // namespace detail
} // namespace as_tuple

template <typename Archive, AsTupleStr T>
inline void save(Archive &ar, const T &v) {
  as_tuple::detail::archive<T>(ar, v.as_tuple());
}

template <typename Archive, AsTupleStr T>
requires(std::copyable<T> &&std::default_initializable<
         AsTupleT<T>>) inline void load(Archive &ar, T &v) {
  AsTupleT<T> tup;
  as_tuple::detail::archive<T>(ar, tup);
  v = T::from_tuple(tup);
}
} // namespace cereal
