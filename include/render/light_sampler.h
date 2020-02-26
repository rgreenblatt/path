#pragma once

namespace render {
enum class LightSamplerType { NoLightSampling, WeightedAABB, RandomTriangle };

template <LightSamplerType type> struct LightSamplerSettings;

template <> struct LightSamplerSettings<LightSamplerType::NoLightSampling> {};

template <> struct LightSamplerSettings<LightSamplerType::WeightedAABB> {};

template <> struct LightSamplerSettings<LightSamplerType::RandomTriangle> {};
}; // namespace render
