#pragma once

namespace render {
// TODO: add more
enum class DirSamplerType { Uniform, BRDF };

template <DirSamplerType type> struct DirSamplerSettings;

template <> struct DirSamplerSettings<DirSamplerType::Uniform> {};

template <> struct DirSamplerSettings<DirSamplerType::BRDF> {};
}; // namespace render
