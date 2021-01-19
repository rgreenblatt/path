#pragma once

#include "lib/settings.h"
#include "meta/all_values_enum.h"
#include "meta/pick_type.h"
#include "meta/predicate_for_all_values.h"
#include "rng/enum_rng/rng_type.h"
#include "rng/sobel/settings.h"
#include "rng/uniform/settings.h"

namespace rng {
namespace enum_rng {
template <RngType type>
using Settings = PickType<type, uniform::Settings, sobel::Settings>;

template <RngType type>
struct SettingsValid : BoolWrapper<Setting<Settings<type>>> {};

static_assert(PredicateForAllValues<RngType>::value<SettingsValid>);
} // namespace enum_rng
} // namespace rng
