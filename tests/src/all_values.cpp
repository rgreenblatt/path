#include "meta/all_values/all_values.h"
#include "meta/all_values/impl/enum.h"
#include "meta/all_values/impl/integral.h"
#include "meta/all_values/impl/pow_2.h"
#include "meta/all_values/impl/range.h"
#include "meta/all_values/impl/tuple.h"
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
