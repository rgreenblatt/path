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
class CompileTimeSettings {
public:
  using AccelType = intersect::accel::AccelType;
  using RngType = rng::RngType;
  using T = std::tuple<AccelType, AccelType, LightSamplerType, DirSamplerType,
                       TermProbType, RngType>;

  constexpr CompileTimeSettings(const T &v) : values_(v) {}

  template <typename... Vals>
  constexpr CompileTimeSettings(Vals &&... vals) : values_({vals...}) {}

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

static_assert(Setting<CompileTimeSettings>);

struct GeneralSettings {
  ComputationSettings settings;
};

// TODO: serialization???
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
      AccelType::LoopAll,
      AccelType::LoopAll,
      LightSamplerType::NoDirectLighting,
      DirSamplerType::Uniform,
      TermProbType::Constant,
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
};

static_assert(Setting<Settings>);
} // namespace render
