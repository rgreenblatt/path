#pragma once

#include "lib/settings.h"

#include <compare>

namespace intersect {
namespace accel {
struct SAHeuristicSettings {
  float traversal_cost;
  float intersection_cost;

  template <typename Archive> void serialize(Archive &archive) {
    archive(NVP(traversal_cost), NVP(intersection_cost));
  }

  ATTR_PURE constexpr bool
  operator==(const SAHeuristicSettings &) const = default;
};

static_assert(Setting<SAHeuristicSettings>);
} // namespace accel
} // namespace intersect
