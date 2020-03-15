#pragma once

#include "lib/settings.h"

namespace render {
enum class LightSamplerType { NoLightSampling, WeightedAABB, RandomTriangle };

template <LightSamplerType type> struct LightSamplerSettings;

template <>
struct LightSamplerSettings<LightSamplerType::NoLightSampling> : EmptySettings {
};

template <>
struct LightSamplerSettings<LightSamplerType::WeightedAABB> : EmptySettings {};

template <>
struct LightSamplerSettings<LightSamplerType::RandomTriangle> : EmptySettings {
};

static_assert(Setting<LightSamplerSettings<LightSamplerType::NoLightSampling>>);
static_assert(Setting<LightSamplerSettings<LightSamplerType::WeightedAABB>>);
static_assert(Setting<LightSamplerSettings<LightSamplerType::RandomTriangle>>);
}; // namespace render
