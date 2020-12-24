#pragma once

#include "integrate/light_sampler/light_sampler.h"
#include "integrate/light_sampler/no_light_sampling/settings.h"

namespace integrate {
namespace light_sampler {
namespace no_light_sampling {
struct NoLightSampling {
public:
  class Ref {
  public:
    HOST_DEVICE Ref() = default;

    HOST_DEVICE Ref(const Settings &) {}

    static constexpr unsigned max_sample_size = 0;
    static constexpr bool performs_samples = false;

    template <bsdf::BSDF B, rng::RngState R>
    HOST_DEVICE LightSamples<max_sample_size>
    operator()(const Eigen::Vector3f &, const bsdf::Material<B> &,
               const Eigen::Vector3f &, const Eigen::Vector3f &, R &) const {
      return {{}, 0};
    }
  };

  template <bsdf::BSDF B, intersect::Object O>
  auto gen(const Settings &settings, Span<const scene::EmissiveCluster>,
           Span<const unsigned>, Span<const bsdf::Material<B>>,
           SpanSized<const intersect::TransformedObject>, Span<const unsigned>,
           Span<const O>) {
    return Ref(settings);
  }
};

static_assert(FullyGeneralLightSampler<NoLightSampling, Settings>);
} // namespace no_light_sampling
} // namespace light_sampler
} // namespace integrate