#pragma once

// TODO: better header name???

#include "intersect/ray.h"
#include "lib/array_vec.h"
#include "lib/float_rgb.h"
#include "lib/optional.h"
#include "lib/tagged_union.h"
#include "meta/all_values/impl/enum.h"

#include <concepts>

namespace integrate {
namespace detail {
template <typename T> struct RayInfo {
  T multiplier;
  std::optional<float> target_distance;
};

template <typename T> struct RayRayInfo {
  intersect::Ray ray;
  RayInfo<T> info;
};
} // namespace detail

using FRayInfo = detail::RayInfo<float>;
using ArrRayInfo = detail::RayInfo<FloatRGB>;
using FRayRayInfo = detail::RayRayInfo<float>;
using ArrRayRayInfo = detail::RayRayInfo<FloatRGB>;

// this should probably be a class with a friend struct...
template <unsigned max_num_light_samples> struct RenderingEquationState {
  HOST_DEVICE static RenderingEquationState
  initial_state(const FRayRayInfo &ray_ray_info) {
    return RenderingEquationState{
        .iters = 0,
        .count_emission = true,
        .has_next_sample = true,
        .ray_ray_info = {ray_ray_info.ray,
                         {FloatRGB::Constant(ray_ray_info.info.multiplier),
                          ray_ray_info.info.target_distance}},
        .light_samples = {},
        .float_rgb_total = FloatRGB::Zero(),
    };
  }

  unsigned iters;
  bool count_emission;
  bool has_next_sample;
  ArrRayRayInfo ray_ray_info;
  ArrayVec<ArrRayInfo, max_num_light_samples> light_samples;
  FloatRGB float_rgb_total;

  HOST_DEVICE unsigned num_samples() const {
    return light_samples.size() + has_next_sample;
  }
};

static_assert(std::semiregular<RenderingEquationState<3>>);

template <unsigned max_num_light_samples>
struct RenderingEquationNextIteration {
  RenderingEquationState<max_num_light_samples> state;
  ArrayVec<intersect::Ray, max_num_light_samples + 1> rays;
};

enum class IterationOutputType {
  NextIteration,
  Finished,
};

template <unsigned max_num_light_samples>
using IterationOutput =
    TaggedUnion<IterationOutputType,
                RenderingEquationNextIteration<max_num_light_samples>,
                FloatRGB>;
} // namespace integrate
