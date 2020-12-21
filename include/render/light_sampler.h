#pragma once

#include "lib/settings.h"

namespace render {
enum class LightSamplerType { NoLightSampling, RandomTriangle };

template <LightSamplerType type> struct LightSamplerSettings;

template <>
struct LightSamplerSettings<LightSamplerType::NoLightSampling> : EmptySettings {
};

template <> struct LightSamplerSettings<LightSamplerType::RandomTriangle> {
  unsigned binary_search_threshold = std::numeric_limits<unsigned>::max();

  template <class Archive> void serialize(Archive &archive) {
    archive(CEREAL_NVP(binary_search_threshold));
  }
};

static_assert(Setting<LightSamplerSettings<LightSamplerType::NoLightSampling>>);
static_assert(Setting<LightSamplerSettings<LightSamplerType::RandomTriangle>>);
}; // namespace render
