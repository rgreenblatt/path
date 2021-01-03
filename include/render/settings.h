#pragma once

#include "integrate/dir_sampler/enum_dir_sampler/settings.h"
#include "integrate/light_sampler/enum_light_sampler/settings.h"
#include "integrate/term_prob/enum_term_prob/settings.h"
#include "intersect/accel/enum_accel/settings.h"
#include "lib/settings.h"
#include "lib/tagged_union.h"
#include "meta/all_values.h"
#include "meta/tag.h"
#include "render/general_settings.h"
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
public:
  TaggedUnionPerInstance<AccelType, enum_accel::Settings> flat_accel = {
      TAG(AccelType::KDTree)};

  TaggedUnionPerInstance<LightSamplerType, enum_light_sampler::Settings>
      light_sampler = {TAG(LightSamplerType::RandomTriangle)};

  TaggedUnionPerInstance<DirSamplerType, enum_dir_sampler::Settings>
      dir_sampler = {TAG(DirSamplerType::BSDF)};

  TaggedUnionPerInstance<TermProbType, enum_term_prob::Settings> term_prob = {
      TAG(TermProbType::MultiplierFunc)};

  TaggedUnionPerInstance<RngType, rng::enum_rng::Settings> rng = {
      TAG(RngType::Sobel)};

  GeneralSettings general_settings;

  template <typename Archive> void serialize(Archive &ar) {
    ar(NVP(flat_accel), NVP(light_sampler), NVP(dir_sampler), NVP(term_prob),
       NVP(rng), NVP(general_settings));
  }

  constexpr bool operator==(const Settings &) const = default;

  struct CompileTime {
    AccelType flat_accel_type;
    LightSamplerType light_sampler_type;
    DirSamplerType dir_sampler_type;
    TermProbType term_prob_type;
    RngType rng_type;

    ATTR_PURE constexpr auto
    operator<=>(const CompileTime &other) const = default;
  };

  CompileTime compile_time() const {
    return {
        .flat_accel_type = flat_accel.type(),
        .light_sampler_type = light_sampler.type(),
        .dir_sampler_type = dir_sampler.type(),
        .term_prob_type = term_prob.type(),
        .rng_type = rng.type(),
    };
  }
};

static_assert(Setting<Settings>);
} // namespace render

template <> struct AllValuesImpl<render::Settings::CompileTime> {
private:
  using AccelType = render::AccelType;
  using DirSamplerType = render::DirSamplerType;
  using LightSamplerType = render::LightSamplerType;
  using TermProbType = render::TermProbType;
  using RngType = render::RngType;

public:
  // compile times don't change much from small constant values to 1...
  static constexpr std::array<render::Settings::CompileTime, 2> values = {{
      {AccelType::KDTree, LightSamplerType::RandomTriangle,
       DirSamplerType::BSDF, TermProbType::MultiplierFunc,
       rng::enum_rng::RngType::Sobel},
      {AccelType::KDTree, LightSamplerType::NoLightSampling,
       DirSamplerType::BSDF, TermProbType::MultiplierFunc,
       rng::enum_rng::RngType::Sobel},
      // {AccelType::LoopAll, LightSamplerType::RandomTriangle,
      //  DirSamplerType::BSDF, TermProbType::MultiplierFunc,
      //  rng::enum_rng::RngType::Sobel},
      // {AccelType::KDTree, LightSamplerType::RandomTriangle,
      //  DirSamplerType::Uniform, TermProbType::MultiplierFunc,
      //  rng::enum_rng::RngType::Sobel},
      // {AccelType::LoopAll, LightSamplerType::RandomTriangle,
      //  DirSamplerType::Uniform, TermProbType::MultiplierFunc,
      //  rng::enum_rng::RngType::Sobel},
  }};
};

static_assert(AllValuesEnumerable<render::Settings::CompileTime>);

// All values are unique
static_assert([] {
  auto values = AllValues<render::Settings::CompileTime>;
  for (unsigned i = 0; i < values.size(); ++i) {
    for (unsigned j = 0; j < i; ++j) {
      if (values[i] == values[j]) {
        return false;
      }
    }
  }

  return true;
}());
