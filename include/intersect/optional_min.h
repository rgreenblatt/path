#pragma once

#include "lib/attribute.h"
#include "lib/optional.h"

#include <algorithm>

namespace intersect {
// useful for getting the closest intersection
template <typename... T>
ATTR_PURE_NDEBUG constexpr auto
optional_min(const std::optional<T> &...values) {
  return optional_fold(
      [](const auto &a, const auto &b) {
        return std::optional(std::min(a, b));
      },
      [](const auto &a) { return a; }, values...);
}
} // namespace intersect
