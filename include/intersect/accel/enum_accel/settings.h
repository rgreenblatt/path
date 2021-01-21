#pragma once

#include "intersect/accel/dir_tree/settings.h"
#include "intersect/accel/enum_accel/accel_type.h"
#include "intersect/accel/kdtree/settings.h"
#include "intersect/accel/loop_all/settings.h"
#include "lib/settings.h"
#include "meta/all_values/impl/enum.h"
#include "meta/all_values/pick_type.h"
#include "meta/all_values/predicate_for_all_values.h"

namespace intersect {
namespace accel {
namespace enum_accel {
template <AccelType type>
using Settings =
    PickType<type, loop_all::Settings, kdtree::Settings, dir_tree::Settings>;

template <AccelType type>
struct SettingsValid : std::bool_constant<Setting<Settings<type>>> {};

static_assert(PredicateForAllValues<AccelType>::value<SettingsValid>);
} // namespace enum_accel
} // namespace accel
} // namespace intersect
