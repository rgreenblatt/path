#pragma once

#include "meta/specialization_of.h"

// TODO: hana tuple compare work around...
// https://github.com/boostorg/hana/issues/478
#include <boost/hana/equal.hpp>
#include <boost/hana/less.hpp>
#include <boost/hana/not_equal.hpp>
#include <boost/hana/size.hpp>
#include <boost/hana/tuple.hpp>

#include <concepts>
#include <utility>

// tuple which is preferred for meta programming. It is a structural type
// which allows it to be used as a template argument (in c++20)
template <typename... T> using MetaTuple = boost::hana::tuple<T...>;

template <typename T>
concept IsMetaTuple = SpecializationOf<T, boost::hana::tuple>;

static_assert(IsMetaTuple<MetaTuple<int>>);
static_assert(IsMetaTuple<MetaTuple<bool>>);
static_assert(!IsMetaTuple<bool>);

template <typename... T> constexpr decltype(auto) make_meta_tuple(T &&...vals) {
  return boost::hana::make_tuple(std::forward<T>(vals)...);
}

template <unsigned idx, SpecializationOf<boost::hana::tuple> T>
constexpr decltype(auto) meta_tuple_at(T &&v) {
  return v[boost::hana::size_c<idx>];
}

template <SpecializationOf<boost::hana::tuple> T>
constexpr auto meta_tuple_size(const T &v) {
  return boost::hana::size(v);
}

template <SpecializationOf<boost::hana::tuple> T>
inline constexpr unsigned meta_tuple_size_v =
    decltype(meta_tuple_size(std::declval<const T &>()))::value;
