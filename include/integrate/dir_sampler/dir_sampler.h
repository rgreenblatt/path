#pragma once

#include "bsdf/bsdf.h"
#include "integrate/dir_sample.h"
#include "lib/settings.h"
#include "meta/mock.h"
#include "rng/rng.h"

#include <Eigen/Core>

namespace integrate {
namespace dir_sampler {
struct Sample {
  BSDFSample sample;
  bool is_discrete;
};

// the multiplier in the sample returned by the dir_sampler is the appropriate
// multiplier for integration. This is: brdf * cos(theta) / prob
// note that cos(theta) = direction.dot(normal) where direction is the outgoing
// direction
//
// handling things this way may improve speed somewhat

template <typename T, typename B>
concept DirSamplerRef = requires(const T &dir_sampler,
                                 const Eigen::Vector3f &position, const B &bsdf,
                                 const Eigen::Vector3f &incoming_dir,
                                 const Eigen::Vector3f &normal,
                                 rng::MockRngState &rng) {
  requires ::bsdf::BSDF<B>;
  requires std::copyable<T>;
  { dir_sampler(position, bsdf, incoming_dir, normal, rng) }
  ->DecaysTo<Sample>;
};

template <typename T, typename S, typename B>
concept DirSampler = requires(T &dir_sampler, const S &settings) {
  requires Setting<S>;
  requires std::movable<T>;
  requires std::default_initializable<T>;

  { dir_sampler.gen(settings) }
  ->DirSamplerRef<B>;
};

template <typename T>
concept ContinuousDirSamplerRef = DirSamplerRef<T, bsdf::MockContinuousBSDF>;

template <typename T, typename S>
concept ContinuousDirSampler = DirSampler<T, S, bsdf::MockContinuousBSDF>;

template <typename T>
concept GeneralDirSamplerRef =
    ContinuousDirSamplerRef<T> &&DirSamplerRef<T, bsdf::MockDiscreteBSDF>
        &&DirSamplerRef<T, bsdf::MockContinuousDiscreteBSDF>;

template <typename T, typename S>
concept GeneralDirSampler =
    ContinuousDirSampler<T, S> &&DirSampler<T, S, bsdf::MockDiscreteBSDF>
        &&DirSampler<T, S, bsdf::MockContinuousDiscreteBSDF>;

struct MockContinuousDirSamplerRef : MockCopyable {
  template <bsdf::ContinuousBSDF B, rng::RngState R>
  Sample operator()(const Eigen::Vector3f &, const B &, const Eigen::Vector3f &,
                    const Eigen::Vector3f &, R &) const;
};

static_assert(ContinuousDirSamplerRef<MockContinuousDirSamplerRef>);
} // namespace dir_sampler
} // namespace integrate
