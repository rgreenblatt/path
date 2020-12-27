#pragma once

#include "meta/all_values.h"
#include "render/compile_time_settings.h"

template <> struct AllValuesImpl<render::CompileTimeSettings> {
private:
  using AccelType = render::AccelType;
  using DirSamplerType = render::DirSamplerType;
  using LightSamplerType = render::LightSamplerType;
  using TermProbType = render::TermProbType;
  using RngType = render::RngType;

public:
  // compile times don't change much from small constant values to 1...
  static constexpr std::array<render::CompileTimeSettings, 1> values = {{
      {AccelType::KDTree, LightSamplerType::RandomTriangle,
       DirSamplerType::BSDF, TermProbType::MultiplierFunc,
       rng::enum_rng::RngType::Sobel},
  }};
};

static_assert(AllValuesEnumerable<render::CompileTimeSettings>);
// All values are unique
static_assert([] {
  auto values = AllValues<render::CompileTimeSettings>;
  for (unsigned i = 0; i < values.size(); ++i) {
    for (unsigned j = 0; j < i; ++j) {
      if (values[i] == values[j]) {
        return false;
      }
    }
  }

  return true;
}());
