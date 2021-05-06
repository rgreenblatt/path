#pragma once

#include "intersect/accel/direction_grid/settings.h"
#include "intersect/accel/enum_accel/accel_type.h"
#include "intersect/accel/loop_all/settings.h"
#include "intersect/accel/naive_partition_bvh/settings.h"
#include "intersect/accel/sbvh/settings.h"
#include "lib/settings.h"
#include "meta/all_values/impl/enum.h"
#include "meta/all_values/pick_type.h"
#include "meta/all_values/predicate_for_all_values.h"

namespace intersect {
namespace accel {
namespace enum_accel {
template <AccelType type>
using Settings =
    PickType<type, loop_all::Settings, naive_partition_bvh::Settings,
             sbvh::Settings, direction_grid::Settings>;

template <AccelType type>
struct SettingsValid : std::bool_constant<Setting<Settings<type>>> {};

static_assert(PredicateForAllValues<AccelType>::value<SettingsValid>);
} // namespace enum_accel
} // namespace accel
} // namespace intersect
