#pragma once

#include "integrate/dir_sampler/enum_dir_sampler/settings.h"
#include "integrate/light_sampler/enum_light_sampler/settings.h"
#include "integrate/term_prob/enum_term_prob/settings.h"
#include "intersect/accel/enum_accel/settings.h"
#include "intersectable_scene/to_bulk.h"
#include "lib/settings.h"
#include "lib/tagged_union.h"
#include "meta/all_values/all_values.h"
#include "meta/all_values/impl/enum.h"
#include "meta/all_values/tag.h"
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

enum class IntersectionApproach {
  MegaKernel,
  StreamingFromGeneral,
};

struct Settings {
  using FlatAccelSettings =
      TaggedUnionPerInstance<AccelType, enum_accel::Settings>;

  struct ToBulkSettings {
    intersectable_scene::ToBulkSettings to_bulk_settings;
    FlatAccelSettings accel;

    AccelType type() const { return accel.type(); }

    SETTING_BODY(ToBulkSettings, to_bulk_settings, accel);
  };

  TaggedUnion<IntersectionApproach, FlatAccelSettings, ToBulkSettings>
      intersection = {TagV<IntersectionApproach::MegaKernel>,
                      TagV<AccelType::KDTree>};

  TaggedUnionPerInstance<LightSamplerType, enum_light_sampler::Settings>
      light_sampler = {TagV<LightSamplerType::RandomTriangle>};

  TaggedUnionPerInstance<DirSamplerType, enum_dir_sampler::Settings>
      dir_sampler = {TagV<DirSamplerType::BSDF>};

  TaggedUnionPerInstance<TermProbType, enum_term_prob::Settings> term_prob = {
      TagV<TermProbType::MultiplierFunc>};

  TaggedUnionPerInstance<RngType, rng::enum_rng::Settings> rng = {
      TagV<RngType::Sobel>};

  GeneralSettings general_settings;

  SETTING_BODY(Settings, intersection, light_sampler, dir_sampler, term_prob,
               rng, general_settings)

  using IntersectionType =
      TaggedUnion<IntersectionApproach, AccelType, AccelType>;

  struct CompileTime {
    IntersectionType intersection_type;
    LightSamplerType light_sampler_type;
    DirSamplerType dir_sampler_type;
    TermProbType term_prob_type;
    RngType rng_type;

    ATTR_PURE constexpr auto
    operator<=>(const CompileTime &other) const = default;
  };

  constexpr CompileTime compile_time() const {
    return {
        .intersection_type =
            intersection.visit_tagged([&](auto tag, const auto &value) {
              return IntersectionType(tag, value.type());
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
