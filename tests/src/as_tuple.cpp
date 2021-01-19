#include "meta/all_values_as_tuple.h"
#include "meta/all_values_integral.h"
#include "meta/all_values_range.h"
#include "meta/as_tuple_str_macro.h"
#include "meta/serialize_as_tuple_str.h"
#include "set_same.h"

#include <cereal/archives/binary.hpp>
#include <gtest/gtest.h>

#include <compare>
#include <sstream>

struct Empty {
  AS_TUPLE_STRUCTURAL(Empty);

  auto operator<=>(const Empty &) const = default;
};

static_assert(AsTuple<Empty>);
static_assert(std::same_as<decltype(Empty{}.as_tuple()), MetaTuple<>>);

struct Single {
  int a;

  bool operator==(const Single &) const = default;

  AS_TUPLE_STRUCTURAL(Single, a);
};

static_assert(AsTuple<Single>);
static_assert(std::same_as<decltype(Single{}.as_tuple()), MetaTuple<int>>);

static_assert(Single{8}.as_tuple() == MetaTuple<int>{8});
static_assert(Single::from_tuple(MetaTuple<int>{8}) == Single{8});
static_assert(Single::from_tuple(MetaTuple<int>{8}) != Single{7});

struct Several {
  int a;
  bool b;
  const void *c;

  bool operator==(const Several &) const = default;

  AS_TUPLE_STRUCTURAL(Several, a, b, c);
};

constexpr std::array<int, 2> address = {};
constexpr const void *ptr = address.data();

static_assert(AsTuple<Several>);
static_assert(std::same_as<decltype(Several{}.as_tuple()),
                           MetaTuple<int, bool, const void *>>);
static_assert(Several::from_tuple(MetaTuple<int, bool, const void *>{
                  8, false, ptr}) == Several{8, false, ptr});
static_assert(Several::from_tuple(MetaTuple<int, bool, const void *>{
                  8, true, ptr}) != Several{7, false, ptr});

struct EmptyStr {
  AS_TUPLE_STR_STRUCTURAL(EmptyStr);
};

static_assert(AsTupleStr<EmptyStr>);
static_assert(EmptyStr{}.as_tuple_strs() == std::array<std::string_view, 0>{});

using namespace std::literals;

struct SingleStr {
  int a;
  AS_TUPLE_STR_STRUCTURAL(SingleStr, a);
};

static_assert(AsTupleStr<SingleStr>);
static_assert(SingleStr{}.as_tuple_strs() == std::array{"a"sv});

struct SeveralStr {
  int a;
  bool b;
  char longer_name;

  bool operator==(const SeveralStr &) const = default;

  AS_TUPLE_STR_STRUCTURAL(SeveralStr, a, b, longer_name);
};

static_assert(AsTupleStr<SeveralStr>);
static_assert(SeveralStr{}.as_tuple_strs() ==
              std::array{"a"sv, "b"sv, "longer_name"sv});

// should serialize identically...
struct SeveralStrSameSerialize {
  int a;
  bool b;
  char longer_name;

  template <typename Archive> void serialize(Archive &ar) {
    ar(CEREAL_NVP(a), CEREAL_NVP(b), CEREAL_NVP(longer_name));
  }
};

TEST(AsTuple, serialize) {
  // empty serialization shouldn't crash and should compile etc...
  {
    std::stringstream ss;
    {
      cereal::BinaryOutputArchive o(ss);
      o(EmptyStr{});
    }

    {
      cereal::BinaryInputArchive i(ss);
      EmptyStr out;
      i(out);
    }
  }

  const SeveralStr in{3, true, 'c'};
  {
    std::stringstream ss;
    {
      cereal::BinaryOutputArchive o(ss);
      o(in);
    }

    {
      cereal::BinaryInputArchive i(ss);
      SeveralStr out;
      i(out);

      EXPECT_EQ(in, out);
    }
  }
  {
    std::stringstream ss;
    {
      cereal::BinaryOutputArchive o(ss);
      o(in);
    }
    {
      cereal::BinaryInputArchive i(ss);
      SeveralStrSameSerialize out;
      i(out);

      EXPECT_EQ(out.a, in.a);
      EXPECT_EQ(out.b, in.b);
      EXPECT_EQ(out.longer_name, in.longer_name);
    }
  }
}

static_assert(AllValuesEnumerable<Empty>);
static_assert(AllValues<Empty>.size() == 0);

struct ContainsUpTo3 {
  UpTo<3> value;

  auto operator<=>(const ContainsUpTo3 &) const = default;

  AS_TUPLE_STRUCTURAL(ContainsUpTo3, value);
};

static_assert(AsTuple<ContainsUpTo3>);
static_assert(AllValuesEnumerable<ContainsUpTo3>);
static_assert(AllValues<ContainsUpTo3>.size() == 3);
static_assert(set_same(AllValues<ContainsUpTo3>,
                       {ContainsUpTo3{0}, ContainsUpTo3{1}, ContainsUpTo3{2}}));

struct SeveralAllValues {
  UpTo<3> up_to_3;
  UpTo<2> up_to_2;
  bool bool_v;

  auto operator<=>(const SeveralAllValues &) const = default;

  AS_TUPLE_STRUCTURAL(SeveralAllValues, up_to_3, up_to_2, bool_v);
};

static_assert(AsTuple<SeveralAllValues>);
static_assert(AllValuesEnumerable<SeveralAllValues>);
static_assert(AllValues<SeveralAllValues>.size() == 12);
static_assert(set_same(AllValues<SeveralAllValues>,
                       {
                           SeveralAllValues{0, 0, false},
                           SeveralAllValues{0, 0, true},
                           SeveralAllValues{0, 1, false},
                           SeveralAllValues{0, 1, true},
                           SeveralAllValues{1, 0, false},
                           SeveralAllValues{1, 0, true},
                           SeveralAllValues{1, 1, false},
                           SeveralAllValues{1, 1, true},
                           SeveralAllValues{2, 0, false},
                           SeveralAllValues{2, 0, true},
                           SeveralAllValues{2, 1, false},
                           SeveralAllValues{2, 1, true},
                       }));
