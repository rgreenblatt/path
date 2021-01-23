#pragma once

#include "lib/assert.h"
#include "meta/all_values/n_tag.h"

namespace sequential_dispatch_detail {
template <unsigned i> constexpr auto v = NTag<i>{};

template <unsigned start, unsigned end, typename F>
requires(start != end) constexpr decltype(auto)
    sequential_dispatch(unsigned index, F &&f) {
  debug_assert_assume(index >= start);
  debug_assert_assume(index < end);

  auto r = [&](auto tag) -> decltype(f(v<0>)) {
    if constexpr (tag + start < end) {
      return f(tag);
    } else {
      unreachable_unchecked();
    }
  };

  switch (index - start) {
  case 0:
    return r(v<0>);
  case 1:
    return r(v<1>);
  case 2:
    return r(v<2>);
  case 3:
    return r(v<3>);
  case 4:
    return r(v<4>);
  case 5:
    return r(v<5>);
  case 6:
    return r(v<6>);
  case 7:
    return r(v<7>);
  case 8:
    return r(v<8>);
  case 9:
    return r(v<9>);
  case 10:
    return r(v<10>);
  case 11:
    return r(v<11>);
  case 12:
    return r(v<12>);
  case 13:
    return r(v<13>);
  case 14:
    return r(v<14>);
  case 15:
    return r(v<15>);
  case 16:
    return r(v<16>);
  case 17:
    return r(v<17>);
  case 18:
    return r(v<18>);
  case 19:
    return r(v<19>);
  case 20:
    return r(v<20>);
  case 21:
    return r(v<21>);
  case 22:
    return r(v<22>);
    return r(v<10>);
  case 11:
    return r(v<11>);
  case 12:
    return r(v<12>);
  case 13:
    return r(v<13>);
  case 14:
    return r(v<14>);
  case 15:
    return r(v<15>);
  case 16:
    return r(v<16>);
  case 17:
    return r(v<17>);
  case 18:
    return r(v<18>);
  case 19:
    return r(v<19>);
  case 20:
    return r(v<20>);
  case 21:
    return r(v<21>);
  case 22:
    return r(v<22>);
  case 23:
    return r(v<23>);
  case 24:
    return r(v<24>);
  case 25:
    return r(v<25>);
  case 26:
    return r(v<26>);
  case 27:
    return r(v<27>);
  case 28:
    return r(v<28>);
  case 29:
    return r(v<29>);
  case 30:
    return r(v<30>);
  case 31:
    return r(v<31>);
  default:
    if constexpr (start + 32 < end) {
      return bench_dispatch_impl<start + 32, end>(index, f);
    }
    unreachable_unchecked();
  };
}
} // namespace sequential_dispatch_detail

template <unsigned size, typename F>
requires(size != 0) constexpr decltype(auto)
    sequential_dispatch(unsigned index, F &&f) {
  if (index >= size) {
    unreachable_unchecked();
  }
  sequential_dispatch_detail::sequential_dispatch<0, size>(index, f);
}
