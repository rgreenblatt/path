#include "meta/all_values.h"
#include "meta/all_values_enum.h"
#include "meta/all_values_integral.h"
#include "meta/all_values_pow_2.h"
#include "meta/all_values_range.h"
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
