#pragma once

#include "lib/settings.h"

namespace render {
// TODO: add more
enum class DirSamplerType { Uniform, BRDF };

template <DirSamplerType type> struct DirSamplerSettings;

template <>
struct DirSamplerSettings<DirSamplerType::Uniform> : EmptySettings {};

template <> struct DirSamplerSettings<DirSamplerType::BRDF> : EmptySettings {};

static_assert(Setting<DirSamplerSettings<DirSamplerType::Uniform>>);
static_assert(Setting<DirSamplerSettings<DirSamplerType::BRDF>>);
}; // namespace render
