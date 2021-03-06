#pragma once

#include "integrate/light_sampler/enum_light_sampler/light_sampler_type.h"
#include "integrate/light_sampler/enum_light_sampler/settings.h"
#include "integrate/light_sampler/no_light_sampling/no_light_sampling.h"
#include "integrate/light_sampler/random_triangle/random_triangle.h"
#include "integrate/light_sampler/triangle_light_sampler.h"
#include "lib/settings.h"
#include "meta/all_values/impl/enum.h"
#include "meta/all_values/pick_type.h"
#include "meta/all_values/predicate_for_all_values.h"

namespace integrate {
namespace light_sampler {
namespace enum_light_sampler {
template <LightSamplerType type, ExecutionModel exec>
using EnumLightSampler = PickType<type, no_light_sampling::NoLightSampling,
                                  random_triangle::RandomTriangle<exec>>;

template <LightSamplerType type, ExecutionModel exec>
struct IsLightSampler : std::bool_constant<GeneralBSDFTriangleLightSampler<
                            EnumLightSampler<type, exec>, Settings<type>>> {};

static_assert(PredicateForAllValues<LightSamplerType,
                                    ExecutionModel>::value<IsLightSampler>);
} // namespace enum_light_sampler
} // namespace light_sampler
} // namespace integrate
