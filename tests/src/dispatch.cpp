#include "meta/n_tag_dispatchable.h"
#include "meta/tag_dispatchable.h"

#include <gtest/gtest.h>

struct UpTo3 {
  template <unsigned idx>
  requires(idx < 3) void operator()(NTag<idx>) {}
};

inline constexpr auto empty = [] {};
inline constexpr auto idx_dispatchable = []<unsigned idx>(NTag<idx>) {};
inline constexpr auto t_tag_bool_dispatchable = []<bool v>(TTag<v>) {};

template <unsigned up_to> struct TTagUpToForUpTo8 {
  template <UpTo<8> idx>
  requires(idx < up_to) void operator()(TTag<idx>) {}
};

inline constexpr auto tag_bool_dispatchable = []<unsigned v>(Tag<bool, v>) {};

inline constexpr auto tag_up_to_8_dispatchable =
    []<unsigned v>(Tag<UpTo<8>, v>) {};

template <unsigned up_to> struct TagUpToForUpTo8 {
  template <unsigned idx>
  requires(idx < up_to) void operator()(Tag<UpTo<8>, idx>) {}
};

TEST(dispatch, NTagDispatchable) {
  static_assert(!NTagDispatchable<1, decltype(empty)>);
  static_assert(NTagDispatchable<3, decltype(idx_dispatchable)>);
  static_assert(NTagDispatchable<3, UpTo3>);
  static_assert(!NTagDispatchable<4, UpTo3>);
}

TEST(dispatch, TTagDispatchable) {
  static_assert(!TTagDispatchable<bool, decltype(empty)>);
  static_assert(!TTagDispatchable<int, decltype(empty)>);
  static_assert(TTagDispatchable<bool, decltype(t_tag_bool_dispatchable)>);
  static_assert(!TTagDispatchable<bool, decltype(idx_dispatchable)>);
  static_assert(!TTagDispatchable<UpTo<8>, TTagUpToForUpTo8<3>>);
  static_assert(!TTagDispatchable<UpTo<8>, TTagUpToForUpTo8<7>>);
  static_assert(TTagDispatchable<UpTo<8>, TTagUpToForUpTo8<8>>);
}

TEST(dispatch, TagDispatchable) {
  static_assert(!TagDispatchable<bool, decltype(empty)>);
  static_assert(!TagDispatchable<int, decltype(empty)>);
  static_assert(TagDispatchable<bool, decltype(tag_bool_dispatchable)>);
  static_assert(!TagDispatchable<bool, decltype(t_tag_bool_dispatchable)>);
  static_assert(!TagDispatchable<bool, decltype(idx_dispatchable)>);
  static_assert(TagDispatchable<UpTo<8>, decltype(tag_up_to_8_dispatchable)>);
  static_assert(!TagDispatchable<UpTo<8>, TagUpToForUpTo8<3>>);
  static_assert(!TagDispatchable<UpTo<8>, TagUpToForUpTo8<7>>);
  static_assert(TagDispatchable<UpTo<8>, TagUpToForUpTo8<8>>);
}
