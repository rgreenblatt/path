#pragma once

#include "bsdf/bsdf.h"
#include "integrate/dir_sampler/dir_sampler.h"
#include "lib/cuda/utils.h"
#include "rng/rng.h"

namespace integrate {
namespace dir_sampler {
template <ContinuousDirSamplerRef T> struct RefFromContinuousSampler {
  T continuous_sampler;

  template <bsdf::BSDF B, rng::RngState R>
  HOST_DEVICE Sample operator()(const Eigen::Vector3f &position, const B &bsdf,
                                const Eigen::Vector3f &incoming_dir,
                                const Eigen::Vector3f &normal, R &r) const {
    if constexpr (B::continuous && B::discrete) {
      if (discrete_prob_check(bsdf, incoming_dir, normal, r)) {
        return {bsdf.discrete_sample(incoming_dir, normal, r), true};
      } else {
        return continuous_sampler(position, bsdf, incoming_dir, normal, r);
      }
    } else if constexpr (B::discrete) {
      return {bsdf.discrete_sample(incoming_dir, normal, r), true};
    } else {
      return continuous_sampler(position, bsdf, incoming_dir, normal, r);
    }
  }

private:
  template <bsdf::BSDF B, rng::RngState R>
  static HOST_DEVICE bool
  discrete_prob_check(const B &bsdf, const Eigen::Vector3f &incoming_dir,
                      const Eigen::Vector3f &normal, R &r) {
    float prob_continuous = bsdf.prob_continuous(incoming_dir, normal);
    if (prob_continuous == 0.f) {
      return true;
    }
    if (prob_continuous == 1.f) {
      return false;
    }
    return prob_continuous < r.next();
  }
};

static_assert(GeneralDirSamplerRef<
              RefFromContinuousSampler<MockContinuousDirSamplerRef>>);
} // namespace dir_sampler
} // namespace integrate
