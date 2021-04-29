#pragma once

#include "intersect/accel/enum_accel/settings.h"
#include "lib/settings.h"
#include "lib/tagged_union.h"

namespace render {
struct IndividuallyIntersectableSettings {
  using FlatAccelSettings =
      TaggedUnionPerInstance<intersect::accel::enum_accel::AccelType,
                             intersect::accel::enum_accel::Settings>;

  // TODO: consider adding more accel types (like 2 layer accel)
  FlatAccelSettings accel;

  using CompileTime = intersect::accel::enum_accel::AccelType;

  constexpr CompileTime compile_time() const { return accel.type(); }

  SETTING_BODY(IndividuallyIntersectableSettings, accel);
};

static_assert(Setting<IndividuallyIntersectableSettings>);
} // namespace render
