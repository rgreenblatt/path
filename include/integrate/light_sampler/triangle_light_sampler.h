#pragma once

#include "integrate/light_sampler/light_sampler.h"
#include "intersect/triangle.h"

namespace integrate {
namespace light_sampler {
template <typename T, typename S, typename B>
concept TriangleLightSampler = LightSampler<T, S, intersect::Triangle, B>;

template <typename T, typename S>
concept GeneralBSDFTriangleLightSampler =
    GeneralBSDFLightSampler<T, S, intersect::Triangle>;
} // namespace light_sampler
} // namespace integrate
