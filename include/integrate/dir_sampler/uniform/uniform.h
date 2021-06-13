#pragma once

#include "bsdf/bsdf.h"
#include "integrate/dir_sampler/dir_sampler.h"
#include "integrate/dir_sampler/ref_from_continuous_sampler.h"
#include "integrate/dir_sampler/uniform/settings.h"
#include "integrate/dir_sampler/uniform_direction_sample.h"
#include "lib/assert.h"
#include "lib/cuda/utils.h"
#include "lib/projection.h"
#include "lib/unit_vector.h"
#include "rng/rng.h"

namespace integrate {
namespace dir_sampler {
namespace uniform {
struct Uniform {
public:
  class Ref {
  public:
    HOST_DEVICE Ref() = default;

    HOST_DEVICE Ref(const Settings &) {}

    template <bsdf::ContinuousBSDF B, rng::RngState R>
    HOST_DEVICE Sample operator()(const Eigen::Vector3f &, const B &bsdf,
                                  const UnitVector &incoming_dir,
                                  const UnitVector &normal, R &rng) const {
      bool need_whole_sphere = !bsdf.is_brdf();
      auto direction = uniform_direction_sample(rng, normal, need_whole_sphere);

      float inv_prob_of_direction = (need_whole_sphere ? 4 : 2) * M_PI;

      return Sample{
          {direction, bsdf.continuous_eval(incoming_dir, direction, normal) *
                          direction->dot(*normal) * inv_prob_of_direction},
          false};
    }
  };

  auto gen(const Settings &settings) {
    return RefFromContinuousSampler<Ref>{Ref(settings)};
  }
};

static_assert(GeneralDirSampler<Uniform, Settings>);
} // namespace uniform
} // namespace dir_sampler
} // namespace integrate
