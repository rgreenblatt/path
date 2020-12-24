#pragma once

#include "bsdf/bsdf.h"
#include "bsdf/material.h"
#include "integrate/dir_sample.h"
#include "intersect/object.h"
#include "intersect/transformed_object.h"
#include "lib/span.h"
#include "rng/rng.h"
#include "scene/emissive_cluster.h"

#include <Eigen/Core>

#include <array>
#include <concepts>

namespace integrate {
namespace light_sampler {
struct LightSample {
  FSample dir_sample;
  float expected_distance;
};

template <unsigned n> struct LightSamples {
  std::array<LightSample, n> samples;
  unsigned num_samples;
};

template <typename T, typename B>
concept LightSamplerRef = requires(const T &light_sampler,
                                   const Eigen::Vector3f &position,
                                   const bsdf::Material<B> &material,
                                   const Eigen::Vector3f &incoming_dir,
                                   const Eigen::Vector3f &normal,
                                   rng::MockRngState &rng) {
  requires bsdf::BSDF<B>;
  requires std::copyable<T>;
  T::max_sample_size;
  T::performs_samples;

  { light_sampler(position, material, incoming_dir, normal, rng) }
  ->DecaysTo<LightSamples<T::max_sample_size>>;
};

// this concept is very dependent on the scene representation...
template <typename T, typename S, typename O, typename B>
concept LightSampler = requires(
    T &light_sampler, const S &settings,
    Span<const scene::EmissiveCluster> emissive_groups,
    Span<const unsigned> emissive_group_ends_per_mesh,
    Span<const bsdf::Material<B>> materials,
    SpanSized<const intersect::TransformedObject> transformed_mesh_objects,
    Span<const unsigned> transformed_mesh_idxs, Span<const O> objects) {
  requires std::default_initializable<T>;
  requires std::movable<T>;
  requires Setting<S>;
  requires intersect::Object<O>;

  {
    light_sampler.gen(settings, emissive_groups, emissive_group_ends_per_mesh,
                      materials, transformed_mesh_objects,
                      transformed_mesh_idxs, objects)
  }
  ->LightSamplerRef<B>;
};

// works for all bsdfs
template <typename T, typename S, typename O>
concept GeneralBSDFLightSampler =
    LightSampler<T, S, O, bsdf::MockContinuousBSDF>
        &&LightSampler<T, S, O, bsdf::MockDiscreteBSDF>
            &&LightSampler<T, S, O, bsdf::MockContinuousDiscreteBSDF>;

// works for all objects
template <typename T, typename S, typename B>
concept GeneralObjectLightSampler =
    LightSampler<T, S, intersect::MockObject, B>;

// works for all objects and bsdfs
template <typename T, typename S>
concept FullyGeneralLightSampler =
    GeneralBSDFLightSampler<T, S, intersect::MockObject>;
} // namespace light_sampler
} // namespace integrate
