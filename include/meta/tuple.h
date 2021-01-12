#pragma once

#include "meta/specialization_of.h"

#include <boost/hana/equal.hpp>
#include <boost/hana/less.hpp>
#include <boost/hana/not_equal.hpp>
#include <boost/hana/tuple.hpp>

#include <concepts>
#include <utility>

// tuple which is preferred for meta programming. It is a structural type
// which allows it to be used as a template argument (in c++20)
template <typename... T> using MetaTuple = boost::hana::tuple<T...>;

template <typename... T> constexpr decltype(auto) make_meta_tuple(T &&...vals) {
  return boost::hana::make_tuple(std::forward<T>(vals)...);
}

template <unsigned idx, SpecializationOf<boost::hana::tuple> T>
constexpr decltype(auto) meta_tuple_at(T &&v) {
  return v[boost::hana::size_c<idx>];
}
