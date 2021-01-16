#include "meta/n_tag_dispatchable.h"
#include "meta/tag_dispatchable.h"
#include "meta/all_values_range.h"
#include "meta/all_values_integral.h"

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
inline constexpr auto any_dispatchable = [](auto) {};
inline constexpr auto return_type_varies = [](auto v) { return v; };

inline constexpr auto tag_up_to_8_dispatchable =
    []<unsigned v>(Tag<UpTo<8>, v>) {};

template <unsigned up_to> struct TagUpToForUpTo8 {
  template <unsigned idx>
  requires(idx < up_to) void operator()(Tag<UpTo<8>, idx>) {}
};

static_assert(!NTagDispatchable<1, decltype(empty)>);
static_assert(NTagDispatchable<3, decltype(idx_dispatchable)>);
static_assert(NTagDispatchable<3, UpTo3>);
static_assert(!NTagDispatchable<4, UpTo3>);
static_assert(NTagDispatchable<3, decltype(any_dispatchable)>);
static_assert(!NTagDispatchable<3, decltype(return_type_varies)>);
static_assert(NTagDispatchable<1, decltype(return_type_varies)>);
static_assert(!NTagDispatchable<0, decltype(any_dispatchable)>);

static_assert(!TagDispatchable<bool, decltype(empty)>);
static_assert(!TagDispatchable<int, decltype(empty)>);
static_assert(TagDispatchable<bool, decltype(tag_bool_dispatchable)>);
static_assert(!TagDispatchable<bool, decltype(t_tag_bool_dispatchable)>);
static_assert(!TagDispatchable<bool, decltype(idx_dispatchable)>);
static_assert(TagDispatchable<UpTo<8>, decltype(tag_up_to_8_dispatchable)>);
static_assert(!TagDispatchable<UpTo<8>, TagUpToForUpTo8<3>>);
static_assert(!TagDispatchable<UpTo<8>, TagUpToForUpTo8<7>>);
static_assert(TagDispatchable<UpTo<8>, TagUpToForUpTo8<8>>);
static_assert(TagDispatchable<UpTo<8>, decltype(any_dispatchable)>);
static_assert(!AllTypesSame<decltype(return_type_varies(NTag<0>{})),
                            decltype(return_type_varies(NTag<1>{}))>);
static_assert(!TagDispatchable<UpTo<8>, decltype(return_type_varies)>);
static_assert(TagDispatchable<UpTo<1>, decltype(return_type_varies)>);
static_assert(!TagDispatchable<UpTo<0>, decltype(any_dispatchable)>);

static_assert(std::same_as<decltype(to_tag(TTag<false>{})), Tag<bool, 0>>);
static_assert(std::same_as<decltype(to_tag(TTag<true>{})), Tag<bool, 1>>);
static_assert(std::same_as<decltype(to_tag<bool>(NTag<0>{})), Tag<bool, 0>>);
static_assert(std::same_as<decltype(to_tag<bool>(NTag<1>{})), Tag<bool, 1>>);
