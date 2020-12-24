#pragma once

#include "bsdf/bsdf.h"
#include "integrate/dir_sampler/bsdf_sampler/settings.h"
#include "integrate/dir_sampler/dir_sampler.h"
#include "integrate/dir_sampler/ref_from_continuous_sampler.h"
#include "lib/cuda/utils.h"
#include "rng/rng.h"

namespace integrate {
namespace dir_sampler {
namespace bsdf_sampler {
struct BSDFSampler {
public:
  class Ref {
  public:
    HOST_DEVICE Ref() = default;

    HOST_DEVICE Ref(const Settings &) {}

    template <bsdf::ContinuousBSDF B, rng::RngState R>
    HOST_DEVICE Sample operator()(const Eigen::Vector3f &,
                                  const B &bsdf_sampler,
                                  const Eigen::Vector3f &direction,
                                  const Eigen::Vector3f &normal, R &rng) const {
      return {bsdf_sampler.continuous_sample(direction, normal, rng), false};
    }
  };

  auto gen(const Settings &settings) {
    return RefFromContinuousSampler<Ref>{Ref(settings)};
  }
};

static_assert(GeneralDirSampler<BSDFSampler, Settings>);
} // namespace bsdf_sampler
} // namespace dir_sampler
} // namespace integrate
