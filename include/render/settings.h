#pragma once

#include "integrate/dir_sampler/enum_dir_sampler/settings.h"
#include "integrate/light_sampler/enum_light_sampler/settings.h"
#include "integrate/rendering_equation_settings.h"
#include "integrate/term_prob/enum_term_prob/settings.h"
#include "lib/settings.h"
#include "lib/tagged_union.h"
#include "meta/all_values/all_values.h"
#include "meta/all_values/impl/enum.h"
#include "meta/all_values/tag.h"
#include "render/kernel_approach_settings.h"
#include "rng/enum_rng/settings.h"

namespace render {
namespace enum_accel = intersect::accel::enum_accel;
namespace enum_dir_sampler = integrate::dir_sampler::enum_dir_sampler;
namespace enum_light_sampler = integrate::light_sampler::enum_light_sampler;
namespace enum_term_prob = integrate::term_prob::enum_term_prob;

using enum_accel::AccelType;
using enum_dir_sampler::DirSamplerType;
using enum_light_sampler::LightSamplerType;
using enum_term_prob::TermProbType;
using rng::enum_rng::RngType;

struct Settings {
  KernelApproachSettings kernel_approach = {tag_v<KernelApproach::MegaKernel>};

  TaggedUnionPerInstance<LightSamplerType, enum_light_sampler::Settings>
      light_sampler = {tag_v<LightSamplerType::RandomTriangle>};

  TaggedUnionPerInstance<DirSamplerType, enum_dir_sampler::Settings>
      dir_sampler = {tag_v<DirSamplerType::BSDF>};

  TaggedUnionPerInstance<TermProbType, enum_term_prob::Settings> term_prob = {
      tag_v<TermProbType::MultiplierFunc>};

  TaggedUnionPerInstance<RngType, rng::enum_rng::Settings> rng = {
      tag_v<RngType::Sobel>};

  integrate::RenderingEquationSettings rendering_equation_settings;

  SETTING_BODY(Settings, kernel_approach, light_sampler, dir_sampler, term_prob,
               rng, rendering_equation_settings)

  struct CompileTime {
    KernelApproachCompileTime kernel_approach_type;
    LightSamplerType light_sampler_type;
    DirSamplerType dir_sampler_type;
    TermProbType term_prob_type;
    RngType rng_type;

    ATTR_PURE constexpr auto
    operator<=>(const CompileTime &other) const = default;
  };

  constexpr CompileTime compile_time() const {
    return {
        .kernel_approach_type = kernel_approach.visit_tagged(
            [&](auto tag, const auto &value) -> KernelApproachCompileTime {
              return {tag, value.compile_time()};
            }),
        .light_sampler_type = light_sampler.type(),
        .dir_sampler_type = dir_sampler.type(),
        .term_prob_type = term_prob.type(),
        .rng_type = rng.type(),
    };
  }
};

static_assert(Setting<Settings>);
} // namespace render
