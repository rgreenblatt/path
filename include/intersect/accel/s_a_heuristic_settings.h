#pragma once

#include "lib/settings.h"

#include <compare>

namespace intersect {
namespace accel {
struct SAHeuristicSettings {
  float traversal_cost;
  float intersection_cost;

  template <class Archive> void serialize(Archive &archive) {
    archive(CEREAL_NVP(traversal_cost), CEREAL_NVP(intersection_cost));
  }

  constexpr inline bool operator==(const SAHeuristicSettings &) const = default;
};

static_assert(Setting<SAHeuristicSettings>);
} // namespace accel
} // namespace intersect
