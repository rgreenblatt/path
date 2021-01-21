#include "meta/all_values/impl/integral.h"
#include "meta/all_values/impl/range.h"
#include "meta/all_values/n_tag_dispatchable.h"
#include "meta/all_values/tag_dispatchable.h"

struct UpTo3 {
  template <unsigned idx>
  requires(idx < 3) void operator()(NTag<idx>) {}
};

inline constexpr auto empty = [] {};
inline constexpr auto idx_dispatchable = []<unsigned idx>(NTag<idx>) {};

inline constexpr auto tag_bool_dispatchable = []<unsigned v>(Tag<bool, v>) {};
inline constexpr auto any_dispatchable = [](auto) {};
inline constexpr auto return_type_varies = [](auto v) { return v; };

inline constexpr auto tag_up_to_8_dispatchable =
    []<unsigned v>(Tag<UpTo<8>, v>) {};

template <unsigned up_to> struct TagUpToForUpTo8 {
  template <unsigned idx>
  requires(idx < up_to) void operator()(Tag<UpTo<8>, idx>) {}
};

// TODO: gcc work around (should work on trunk)
#ifdef __clang__
static_assert(!NTagDispatchable<decltype(empty), 1>);
static_assert(NTagDispatchable<decltype(idx_dispatchable), 3>);
static_assert(NTagDispatchable<UpTo3, 3>);
static_assert(!NTagDispatchable<UpTo3, 4>);
static_assert(NTagDispatchable<decltype(any_dispatchable), 3>);
static_assert(!NTagDispatchable<decltype(return_type_varies), 3>);
static_assert(NTagDispatchable<decltype(return_type_varies), 1>);
static_assert(!NTagDispatchable<decltype(any_dispatchable), 0>);

static_assert(!TagDispatchable<decltype(empty), bool>);
static_assert(!TagDispatchable<decltype(empty), int>);
static_assert(TagDispatchable<decltype(tag_bool_dispatchable), bool>);
static_assert(!TagDispatchable<decltype(idx_dispatchable), bool>);
static_assert(TagDispatchable<decltype(tag_up_to_8_dispatchable), UpTo<8>>);
static_assert(!TagDispatchable<TagUpToForUpTo8<3>, UpTo<8>>);
static_assert(!TagDispatchable<TagUpToForUpTo8<7>, UpTo<8>>);
static_assert(TagDispatchable<TagUpToForUpTo8<8>, UpTo<8>>);
static_assert(TagDispatchable<decltype(any_dispatchable), UpTo<8>>);
static_assert(!AllTypesSame<decltype(return_type_varies(NTag<0>{})),
                            decltype(return_type_varies(NTag<1>{}))>);
static_assert(!TagDispatchable<decltype(return_type_varies), UpTo<8>>);
static_assert(TagDispatchable<decltype(return_type_varies), UpTo<1>>);
static_assert(!TagDispatchable<decltype(any_dispatchable), UpTo<0>>);
#endif

static_assert(std::same_as<decltype(to_tag<bool>(NTag<0>{})), Tag<bool, 0>>);
static_assert(std::same_as<decltype(to_tag<bool>(NTag<1>{})), Tag<bool, 1>>);
