#pragma once

#include "integrate/dir_sampler/bsdf_sampler/settings.h"
#include "integrate/dir_sampler/enum_dir_sampler/dir_sampler_type.h"
#include "integrate/dir_sampler/uniform/settings.h"
#include "lib/settings.h"
#include "meta/all_values/impl/enum.h"
#include "meta/all_values/pick_type.h"
#include "meta/all_values/predicate_for_all_values.h"

namespace integrate {
namespace dir_sampler {
namespace enum_dir_sampler {
template <DirSamplerType type>
using Settings = PickType<type, uniform::Settings, bsdf_sampler::Settings>;

template <DirSamplerType type>
struct SettingsValid : std::bool_constant<Setting<Settings<type>>> {};

static_assert(PredicateForAllValues<DirSamplerType>::value<SettingsValid>);
} // namespace enum_dir_sampler
} // namespace dir_sampler
} // namespace integrate
