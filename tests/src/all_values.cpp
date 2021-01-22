#include "meta/all_values/all_values.h"
#include "meta/all_values/impl/enum.h"
#include "meta/all_values/impl/integral.h"
#include "meta/all_values/impl/pow_2.h"
#include "meta/all_values/impl/range.h"
#include "meta/all_values/impl/tuple.h"
#include "meta/all_values/impl/variant.h"
#include "meta/mock.h"
#include "set_same.h"

#include <algorithm>

enum class EnumTypes {
  A,
  B,
};

static_assert(set_same(AllValues<EnumTypes>, {EnumTypes::A, EnumTypes::B}));

static_assert(!std::default_initializable<UpTo<0>>);
static_assert(!std::default_initializable<Range<3, 3>>);
static_assert(set_same(AllValues<UpTo<0>>, {}));
static_assert(set_same(AllValues<UpTo<1>>, {0}));
static_assert(set_same(AllValues<UpTo<5>>, {0, 1, 2, 3, 4}));
static_assert(set_same(AllValues<Range<1, 2>>, {1}));
static_assert(set_same(AllValues<Range<4, 7>>, {4, 5, 6}));

static_assert(set_same(AllValues<bool>, {false, true}));

static_assert(set_same(AllValues<Pow2<1, 1>>, {1}));
static_assert(set_same(AllValues<Pow2<1, 4>>, {1, 2, 4}));
static_assert(set_same(AllValues<Pow2<2, 4>>, {2, 4}));
static_assert(set_same(AllValues<Pow2InclusiveUpTo<2>>, {1, 2}));

static_assert(AllValues<MetaTuple<>>.size() == 1);
static_assert(set_same(AllValues<MetaTuple<>>, {MetaTuple<>{}}));
static_assert(set_same(AllValues<MetaTuple<bool, bool>>,
                       {
                           MetaTuple<bool, bool>{false, false},
                           MetaTuple<bool, bool>{false, true},
                           MetaTuple<bool, bool>{true, false},
                           MetaTuple<bool, bool>{true, true},
                       }));

static_assert(set_same(AllValues<std::variant<UpTo<0>>>, {}));
static_assert(set_same(AllValues<std::variant<UpTo<1>>>, {{0}}));
static_assert(set_same(AllValues<std::variant<UpTo<3>>>, {{0, 1, 2}}));
static_assert(
    set_same(AllValues<std::variant<UpTo<3>, UpTo<2>>>,
             {
                 std::variant<UpTo<3>, UpTo<2>>{std::in_place_index<0>, 0},
                 std::variant<UpTo<3>, UpTo<2>>{std::in_place_index<0>, 1},
                 std::variant<UpTo<3>, UpTo<2>>{std::in_place_index<0>, 2},
                 std::variant<UpTo<3>, UpTo<2>>{std::in_place_index<1>, 0},
                 std::variant<UpTo<3>, UpTo<2>>{std::in_place_index<1>, 1},
             }));
static_assert(
    set_same(AllValues<std::variant<Pow2<1, 4>, UpTo<3>>>,
             {
                 std::variant<Pow2<1, 4>, UpTo<3>>{std::in_place_index<0>, 1},
                 std::variant<Pow2<1, 4>, UpTo<3>>{std::in_place_index<0>, 2},
                 std::variant<Pow2<1, 4>, UpTo<3>>{std::in_place_index<0>, 4},
                 std::variant<Pow2<1, 4>, UpTo<3>>{std::in_place_index<1>, 0},
                 std::variant<Pow2<1, 4>, UpTo<3>>{std::in_place_index<1>, 1},
                 std::variant<Pow2<1, 4>, UpTo<3>>{std::in_place_index<1>, 2},
             }));

template <> struct AllValuesImpl<MockMovable> {
  static constexpr std::array<MockMovable, 0> values = {};
};

static_assert(!AllValuesEnumerable<MockMovable>);

struct NoCompare {};

template <> struct AllValuesImpl<NoCompare> {
  static constexpr std::array<NoCompare, 0> values = {};
};

static_assert(!AllValuesEnumerable<NoCompare>);

struct NoImpl : UpTo<5> {
  using UpTo<5>::UpTo;
};

static_assert(!AllValuesEnumerable<NoImpl>);

struct NotSorted : UpTo<5> {
  using UpTo<5>::UpTo;
};

template <> struct AllValuesImpl<NotSorted> {
  static constexpr auto values = std::array<NotSorted, 4>{1, 2, 4, 3};
};

struct NotUnique : UpTo<5> {
  using UpTo<5>::UpTo;
};

template <> struct AllValuesImpl<NotUnique> {
  static constexpr auto values = std::array<NotSorted, 4>{1, 2, 4, 4};
};

static_assert(!AllValuesEnumerable<NotSorted>);
