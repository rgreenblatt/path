#pragma once

namespace render {
// TODO: add more
enum class LightSamplerType { NoDirectLighting, WeightedAABB };

template <LightSamplerType type> struct LightSamplerSettings;

template <> struct LightSamplerSettings<LightSamplerType::NoDirectLighting> {};

template <> struct LightSamplerSettings<LightSamplerType::WeightedAABB> {
  // TODO
};
}; // namespace render
