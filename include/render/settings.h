#pragma once

#include "compile_time_dispatch/one_per_instance.h"
#include "intersect/accel/accel.h"
#include "lib/settings.h"
#include "render/computation_settings.h"
#include "render/dir_sampler.h"
#include "render/light_sampler.h"
#include "render/term_prob.h"
#include "rng/rng.h"

#include <tuple>

namespace render {
using CompileTimeSettingsFull =
    std::tuple<intersect::accel::AccelType, intersect::accel::AccelType,
               LightSamplerType, DirSamplerType, TermProbType, rng::RngType>;

struct CompileTimeSettingsSubset : public CompileTimeSettingsFull {};
struct CompileTimeSettingsAll : public CompileTimeSettingsFull {};
} // namespace render

template <>
struct CompileTimeDispatchableImpl<render::CompileTimeSettingsSubset> {
private:
  using AccelType = intersect::accel::AccelType;
  using RngType = rng::RngType;
  using LightSamplerType = render::LightSamplerType;
  using DirSamplerType = render::DirSamplerType;
  using TermProbType = render::TermProbType;

public:
  static constexpr std::array<render::CompileTimeSettingsSubset, 5> values = {{
      {{AccelType::KDTree, AccelType::KDTree, LightSamplerType::RandomTriangle,
        DirSamplerType::Uniform, TermProbType::MultiplierFunc,
        rng::RngType::Uniform}},
      {{AccelType::KDTree, AccelType::KDTree, LightSamplerType::RandomTriangle,
        DirSamplerType::BRDF, TermProbType::MultiplierFunc,
        rng::RngType::Uniform}},
  }};
};

static_assert(CompileTimeDispatchable<render::CompileTimeSettingsSubset>);

template <>
struct CompileTimeDispatchableImpl<render::CompileTimeSettingsAll>
    : public CompileTimeDispatchableImpl<render::CompileTimeSettingsFull> {};

namespace cereal {
template <class Archive, Enum T>
inline std::string save_minimal(Archive const &, T const &enum_v) {
  return std::string(magic_enum::enum_name(enum_v));
}

template <class Archive, Enum T>
inline void load_minimal(Archive const &, T &enum_v, const std::string &s) {
  auto val_op = magic_enum::enum_cast<T>(s);
  if (val_op.has_value()) {
    enum_v = val_op.value();
  } else {
    std::cerr << "failed to load enum with string: " << s << std::endl;
  }
}
} // namespace cereal

namespace render {
class CompileTimeSettings {
public:
  using AccelType = intersect::accel::AccelType;
  using RngType = rng::RngType;
  using T =
#ifdef QUICK_BUILD
      CompileTimeSettingsSubset
#else
      CompileTimeSettingsFull
#endif
      ;

  constexpr CompileTimeSettings(const T &v) : values_(v) {}

  template <typename... Vals>
  constexpr CompileTimeSettings(Vals &&... vals) : values_({{vals...}}) {}

  constexpr AccelType triangle_accel_type() const {
    return std::get<0>(values_);
  }

  AccelType &triangle_accel_type() { return std::get<0>(values_); }

  constexpr AccelType mesh_accel_type() const { return std::get<1>(values_); }

  AccelType &mesh_accel_type() { return std::get<1>(values_); }

  constexpr LightSamplerType light_sampler_type() const {
    return std::get<2>(values_);
  }

  LightSamplerType &light_sampler_type() { return std::get<2>(values_); }

  constexpr DirSamplerType dir_sampler_type() const {
    return std::get<3>(values_);
  }

  DirSamplerType &dir_sampler_type() { return std::get<3>(values_); }

  constexpr TermProbType term_prob_type() const { return std::get<4>(values_); }

  TermProbType &term_prob_type() { return std::get<4>(values_); }

  constexpr RngType rng_type() const { return std::get<5>(values_); }

  RngType &rng_type() { return std::get<5>(values_); }

  constexpr const T &values() const { return values_; }

private:
  T values_;
};

static_assert(CompileTimeDispatchable<CompileTimeSettings::T>);

struct GeneralSettings {
  ComputationSettings computation_settings;

  template <class Archive> void serialize(Archive &archive) {
    archive(CEREAL_NVP(computation_settings));
  }
};

static_assert(Setting<GeneralSettings>);

struct Settings {
private:
  // default should be pretty reasonable...
  using AccelType = intersect::accel::AccelType;
  using RngType = rng::RngType;

  template <AccelType type>
  using AccelSettings = typename intersect::accel::AccelSettings<type>;

  using AllAccelSettings = OnePerInstance<AccelType, AccelSettings>;

  using AllLightSamplerSettings =
      OnePerInstance<LightSamplerType, LightSamplerSettings>;

  using AllDirSamplerSettings =
      OnePerInstance<DirSamplerType, DirSamplerSettings>;

  using AllTermProbSettings = OnePerInstance<TermProbType, TermProbSettings>;

  using AllRngSettings = OnePerInstance<RngType, rng::RngSettings>;

  const CompileTimeSettings default_compile_time = {
      AccelType::KDTree,
      AccelType::KDTree,
      LightSamplerType::RandomTriangle,
      DirSamplerType::BRDF,
      TermProbType::MultiplierFunc,
      RngType::Uniform};

public:
  AllAccelSettings triangle_accel;

  AllAccelSettings mesh_accel;

  AllLightSamplerSettings light_sampler;

  AllDirSamplerSettings dir_sampler;

  AllTermProbSettings term_prob;

  AllRngSettings rng;

  CompileTimeSettings compile_time = default_compile_time;

  GeneralSettings general_settings;

  template <class Archive> void serialize(Archive &archive) {
    auto serialize_item = [&](const auto &name, auto &type, auto &settings) {
      archive(cereal::make_nvp(name, type));
      settings.visit(
          [&](auto &item) {
            archive(cereal::make_nvp(std::string(name) + "_settings", item));
          },
          type);
    };

    serialize_item("triangle_accel", compile_time.triangle_accel_type(),
                   triangle_accel);
    serialize_item("mesh_accel", compile_time.mesh_accel_type(), mesh_accel);
    serialize_item("light_sampler", compile_time.light_sampler_type(),
                   light_sampler);
    serialize_item("dir_sampler", compile_time.dir_sampler_type(), dir_sampler);
    serialize_item("term_prob", compile_time.term_prob_type(), term_prob);
    serialize_item("rng", compile_time.rng_type(), rng);
    archive(CEREAL_NVP(general_settings));
  }
};

static_assert(Setting<Settings>);
} // namespace render
