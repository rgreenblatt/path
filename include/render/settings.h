#pragma once

#include "integrate/dir_sampler/enum_dir_sampler/settings.h"
#include "integrate/light_sampler/enum_light_sampler/settings.h"
#include "integrate/term_prob/enum_term_prob/settings.h"
#include "intersect/accel/enum_accel/settings.h"
#include "lib/settings.h"
#include "meta/all_values.h"
#include "meta/one_per_instance.h"
#include "render/general_settings.h"
#include "rng/enum_rng/rng_type.h"
#include "rng/enum_rng/settings.h"

#include <tuple>

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

using CompileTimeSettingsFull =
    std::tuple<AccelType, LightSamplerType, DirSamplerType, TermProbType,
               RngType>;

struct CompileTimeSettingsSubset : public CompileTimeSettingsFull {
  using CompileTimeSettingsFull::CompileTimeSettingsFull;
};
} // namespace render

template <> struct AllValuesImpl<render::CompileTimeSettingsSubset> {
private:
  using AccelType = render::AccelType;
  using DirSamplerType = render::DirSamplerType;
  using LightSamplerType = render::LightSamplerType;
  using TermProbType = render::TermProbType;
  using RngType = render::RngType;

public:
  // compile times don't change much from small constant values to 1...
  static constexpr std::array<render::CompileTimeSettingsSubset, 2> values = {{
      {AccelType::KDTree, LightSamplerType::RandomTriangle,
       DirSamplerType::BRDF, TermProbType::MultiplierFunc,
       rng::enum_rng::RngType::Uniform},
      {AccelType::KDTree, LightSamplerType::RandomTriangle,
       DirSamplerType::BRDF, TermProbType::MultiplierFunc,
       rng::enum_rng::RngType::Sobel},
  }};
};

static_assert(AllValuesEnumerable<render::CompileTimeSettingsSubset>);
// All values are unique
static_assert([] {
  auto values = AllValues<render::CompileTimeSettingsSubset>;
  for (unsigned i = 0; i < values.size(); ++i) {
    for (unsigned j = 0; j < i; ++j) {
      if (values[i] == values[j]) {
        return false;
      }
    }
  }

  return true;
}());

namespace cereal {
template <typename Archive, Enum T>
inline std::string save_minimal(Archive const &, T const &enum_v) {
  return std::string(magic_enum::enum_name(enum_v));
}

template <typename Archive, Enum T>
inline void load_minimal(Archive const &, T &enum_v, const std::string &s) {
  auto val_op = magic_enum::enum_cast<T>(s);
  if (val_op.has_value()) {
    enum_v = val_op.value();
  } else {
    std::cerr << "failed to load enum with string: " << s << std::endl;
    abort();
  }
}
} // namespace cereal

namespace render {
class CompileTimeSettings {
public:
  using T = CompileTimeSettingsSubset;

  constexpr CompileTimeSettings(const T &v) : values_(v) {}

  template <typename... Vals>
  constexpr CompileTimeSettings(Vals &&...vals) : values_({vals...}) {}

  constexpr AccelType flat_accel_type() const { return std::get<0>(values_); }

  AccelType &flat_accel_type() { return std::get<0>(values_); }

  constexpr LightSamplerType light_sampler_type() const {
    return std::get<1>(values_);
  }

  LightSamplerType &light_sampler_type() { return std::get<1>(values_); }

  constexpr DirSamplerType dir_sampler_type() const {
    return std::get<2>(values_);
  }

  DirSamplerType &dir_sampler_type() { return std::get<2>(values_); }

  constexpr TermProbType term_prob_type() const { return std::get<3>(values_); }

  TermProbType &term_prob_type() { return std::get<3>(values_); }

  constexpr RngType rng_type() const { return std::get<4>(values_); }

  RngType &rng_type() { return std::get<4>(values_); }

  constexpr const T &values() const { return values_; }

  constexpr bool operator==(const CompileTimeSettings &other) const = default;

private:
  T values_;
};

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
      AccelType::KDTree, LightSamplerType::RandomTriangle, DirSamplerType::BRDF,
      TermProbType::MultiplierFunc, RngType::Uniform};

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

    serialize_item("flat_accel", compile_time.flat_accel_type(), flat_accel);
    serialize_item("light_sampler", compile_time.light_sampler_type(),
                   light_sampler);
    serialize_item("dir_sampler", compile_time.dir_sampler_type(), dir_sampler);
    serialize_item("term_prob", compile_time.term_prob_type(), term_prob);
    serialize_item("rng", compile_time.rng_type(), rng);
    archive(NVP(general_settings));
  }

  constexpr bool operator==(const Settings &) const = default;
};

static_assert(Setting<Settings>);
} // namespace render
