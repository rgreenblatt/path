#pragma once

#include "lib/attribute.h"
#include "lib/optional.h"

#include <algorithm>

namespace intersect {
// useful for getting the closest intersection
template <std::movable... T>
requires(AllTypesSame<T...> && sizeof...(T) != 0 &&
         requires(const PackElement<0, T...> &v) {
           std::min(v, v);
         }) ATTR_PURE_NDEBUG
    constexpr auto optional_min(std::optional<T>... values) {
  return optional_fold(
      [](auto a, auto b) {
        // writing min is needed because min doesn't allow moving out...
        if (a < b) {
          return a;
        } else {
          return b;
        }
      },
      std::move(values)...);
}
} // namespace intersect
