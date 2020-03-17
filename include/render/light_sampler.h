#pragma once

#include "lib/settings.h"

namespace render {
enum class LightSamplerType { NoLightSampling, RandomTriangle };

template <LightSamplerType type> struct LightSamplerSettings;

template <>
struct LightSamplerSettings<LightSamplerType::NoLightSampling> : EmptySettings {
};

template <>
struct LightSamplerSettings<LightSamplerType::RandomTriangle> : EmptySettings {
};

static_assert(Setting<LightSamplerSettings<LightSamplerType::NoLightSampling>>);
static_assert(Setting<LightSamplerSettings<LightSamplerType::RandomTriangle>>);
}; // namespace render
