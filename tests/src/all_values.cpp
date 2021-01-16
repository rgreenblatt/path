#include "meta/all_values.h"
#include "meta/all_values_enum.h"
#include "meta/all_values_pow_2.h"
#include "meta/all_values_range.h"
#include "meta/all_values_integral.h"

#include <algorithm>

template <typename T, std::size_t size, typename O>
requires(std::convertible_to<O, T> ||
         size == 0) constexpr bool set_same(std::array<T, size> l,
                                            std::array<O, size> r) {
  std::sort(l.begin(), l.end());
  std::sort(r.begin(), r.end());

  if constexpr (size != 0) {
    for (unsigned i = 0; i < size; ++i) {
      if (l[i] != T(r[i])) {
        return false;
      }
    }
  }

  return true;
}

enum class EnumTypes {
  A,
  B,
};

static_assert(set_same(AllValues<EnumTypes>,
                       std::array{EnumTypes::A, EnumTypes::B}));

static_assert(!std::default_initializable<UpTo<0>>);
static_assert(!std::default_initializable<Range<3, 3>>);
static_assert(set_same(AllValues<UpTo<0>>, std::array<unsigned, 0>{}));
static_assert(set_same(AllValues<UpTo<1>>, std::array{0}));
static_assert(set_same(AllValues<UpTo<5>>, std::array{0, 1, 2, 3, 4}));
static_assert(set_same(AllValues<Range<1, 2>>, std::array{1}));
static_assert(set_same(AllValues<Range<4, 7>>, std::array{4, 5, 6}));

static_assert(set_same(AllValues<bool>, std::array<bool, 2>{false, true}));

static_assert(set_same(AllValues<Pow2<1, 1>>, std::array{1}));
static_assert(set_same(AllValues<Pow2<1, 4>>, std::array{1, 2, 4}));
static_assert(set_same(AllValues<Pow2<2, 4>>, std::array{2, 4}));
static_assert(set_same(AllValues<Pow2InclusiveUpTo<2>>, std::array{1, 2}));
