#pragma once

#include "integrate/light_sampler/enum_light_sampler/light_sampler_type.h"
#include "integrate/light_sampler/no_light_sampling/settings.h"
#include "integrate/light_sampler/random_triangle/settings.h"
#include "lib/settings.h"
#include "meta/all_values_enum.h"
#include "meta/pick_type.h"
#include "meta/predicate_for_all_values.h"

namespace integrate {
namespace light_sampler {
namespace enum_light_sampler {
template <LightSamplerType type>
struct Settings : public PickType<type, no_light_sampling::Settings,
                                  random_triangle::Settings> {};

template <LightSamplerType type>
struct SettingsValid : BoolWrapper<Setting<Settings<type>>> {};

static_assert(PredicateForAllValues<LightSamplerType>::value<SettingsValid>);
} // namespace enum_light_sampler
} // namespace light_sampler
} // namespace integrate
