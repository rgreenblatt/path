#pragma once

#include "integrate/dir_sampler/bsdf_sampler/bsdf_sampler.h"
#include "integrate/dir_sampler/dir_sampler.h"
#include "integrate/dir_sampler/enum_dir_sampler/dir_sampler_type.h"
#include "integrate/dir_sampler/enum_dir_sampler/settings.h"
#include "integrate/dir_sampler/uniform/uniform.h"
#include "lib/settings.h"
#include "meta/pick_type.h"
#include "meta/predicate_for_all_values.h"

namespace integrate {
namespace dir_sampler {
namespace enum_dir_sampler {
template <DirSamplerType type>
struct EnumDirSampler : public PickType<DirSamplerType, type, uniform::Uniform,
                                        bsdf_sampler::BSDFSampler> {};

template <DirSamplerType type>
struct IsGeneralDirSampler
    : BoolWrapper<GeneralDirSampler<EnumDirSampler<type>, Settings<type>>> {};

static_assert(
    PredicateForAllValues<DirSamplerType>::value<IsGeneralDirSampler>);
} // namespace enum_dir_sampler
} // namespace dir_sampler
} // namespace integrate
