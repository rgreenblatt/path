#pragma once

// TODO: better header name???

#include "intersect/ray.h"
#include "lib/array_vec.h"
#include "lib/optional.h"
#include "lib/tagged_union.h"

#include <Eigen/Core>

#include <concepts>

namespace integrate {
namespace detail {
template <typename T> struct RayInfo {
  T multiplier;
  Optional<float> target_distance;
};

template <typename T> struct RayRayInfo {
  intersect::Ray ray;
  RayInfo<T> info;
};
} // namespace detail

using FRayInfo = detail::RayInfo<float>;
using ArrRayInfo = detail::RayInfo<Eigen::Array3f>;
using FRayRayInfo = detail::RayRayInfo<float>;
using ArrRayRayInfo = detail::RayRayInfo<Eigen::Array3f>;

// this should probably be a class with a friend struct...
template <unsigned n_light_samples> struct RenderingEquationState {
  HOST_DEVICE static RenderingEquationState
  initial_state(const FRayRayInfo &ray_ray_info) {
    return RenderingEquationState{
        .iters = 0,
        .count_emission = true,
        .has_next_sample = true,
        .ray_ray_info = {ray_ray_info.ray,
                         {Eigen::Array3f::Constant(
                              ray_ray_info.info.multiplier),
                          ray_ray_info.info.target_distance}},
        .light_samples = {},
        .intensity = Eigen::Array3f::Zero(),
    };
  }

  unsigned iters;
  bool count_emission;
  bool has_next_sample;
  ArrRayRayInfo ray_ray_info;
  ArrayVec<ArrRayInfo, n_light_samples> light_samples;
  Eigen::Array3f intensity;
};

static_assert(std::semiregular<RenderingEquationState<3>>);

template <unsigned n_light_samples> struct RenderingEquationNextIteration {
  RenderingEquationState<n_light_samples> state;
  ArrayVec<intersect::Ray, n_light_samples + 1> rays;
};

enum class IterationOutputType {
  NextIteration,
  Finished,
};

template <unsigned n_light_samples>
using IterationOutput =
    TaggedUnion<IterationOutputType,
                RenderingEquationNextIteration<n_light_samples>,
                Eigen::Array3f>;
} // namespace integrate
