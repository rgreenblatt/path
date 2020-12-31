#pragma once

#include "integrate/dir_sampler/enum_dir_sampler/settings.h"
#include "integrate/light_sampler/enum_light_sampler/settings.h"
#include "integrate/term_prob/enum_term_prob/settings.h"
#include "intersect/accel/enum_accel/settings.h"
#include "lib/one_per_instance.h"
#include "lib/settings.h"
#include "meta/all_values.h"
#include "render/compile_time_settings.h"
#include "render/general_settings.h"
#include "rng/enum_rng/settings.h"

#include <string>

namespace render {
struct Settings {
private:
  using AllAccelSettings = OnePerInstance<AccelType, enum_accel::Settings>;

  using AllLightSamplerSettings =
      OnePerInstance<LightSamplerType, enum_light_sampler::Settings>;

  using AllDirSamplerSettings =
      OnePerInstance<DirSamplerType, enum_dir_sampler::Settings>;

  using AllTermProbSettings =
      OnePerInstance<TermProbType, enum_term_prob::Settings>;

  using AllRngSettings = OnePerInstance<RngType, rng::enum_rng::Settings>;

  static constexpr CompileTimeSettings default_compile_time = {
      AccelType::KDTree, LightSamplerType::RandomTriangle, DirSamplerType::BSDF,
      TermProbType::MultiplierFunc, RngType::Sobel};

public:
  AllAccelSettings flat_accel;

  AllLightSamplerSettings light_sampler;

  AllDirSamplerSettings dir_sampler;

  AllTermProbSettings term_prob;

  AllRngSettings rng;

  CompileTimeSettings compile_time = default_compile_time;

  GeneralSettings general_settings;

  template <typename Archive> void serialize(Archive &archive) {
    auto serialize_item = [&](const auto &name, auto &type, auto &settings) {
      archive(::cereal::make_nvp(name, type));
      settings.visit(
          [&](auto &item) {
            archive(::cereal::make_nvp(
                (std::string(name) + "_settings").c_str(), item));
          },
          type);
    };

    serialize_item("flat_accel", compile_time.flat_accel_type, flat_accel);
    serialize_item("light_sampler", compile_time.light_sampler_type,
                   light_sampler);
    serialize_item("dir_sampler", compile_time.dir_sampler_type, dir_sampler);
    serialize_item("term_prob", compile_time.term_prob_type, term_prob);
    serialize_item("rng", compile_time.rng_type, rng);
    archive(NVP(general_settings));
  }

  constexpr bool operator==(const Settings &) const = default;
};

static_assert(Setting<Settings>);
} // namespace render
