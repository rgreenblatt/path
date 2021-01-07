#pragma once

#include "render/settings.h"

template <> struct AllValuesImpl<render::Settings::CompileTime> {
private:
  using AccelType = render::AccelType;
  using DirSamplerType = render::DirSamplerType;
  using LightSamplerType = render::LightSamplerType;
  using TermProbType = render::TermProbType;
  using RngType = render::RngType;
  using IntersectionApproach = render::IntersectionApproach;
  using IntersectionType = render::Settings::IntersectionType;

public:
  // compile times don't change much from small constant values to 1...
  // compile times do substantially increase for large number of possibilities
#if 1
  constexpr static std::array<render::Settings::CompileTime, 1> values = {{
    {IntersectionType(TAG(IntersectionApproach::MegaKernel), AccelType::KDTree),
     LightSamplerType::RandomTriangle, DirSamplerType::BSDF,
     TermProbType::MultiplierFunc, RngType::Sobel},
    // {IntersectionType(TAG(IntersectionApproach::MegaKernel),
    //                   AccelType::KDTree),
    //  LightSamplerType::NoLightSampling, DirSamplerType::BSDF,
    //  TermProbType::MultiplierFunc, RngType::Sobel},
    // {IntersectionType(TAG(IntersectionApproach::MegaKernel),
    //                   AccelType::KDTree),
    //  LightSamplerType::RandomTriangle, DirSamplerType::Uniform,
    //  TermProbType::MultiplierFunc, RngType::Sobel},
    // {IntersectionType(TAG(IntersectionApproach::MegaKernel),
    //                   AccelType::KDTree),
    //  LightSamplerType::NoLightSampling, DirSamplerType::Uniform,
    //  TermProbType::MultiplierFunc, RngType::Sobel},
    // {IntersectionType(TAG(IntersectionApproach::StreamingFromGeneral),
    //                   AccelType::KDTree),
    //  LightSamplerType::RandomTriangle, DirSamplerType::BSDF,
    //  TermProbType::MultiplierFunc, RngType::Sobel},
    // {IntersectionType(TAG(IntersectionApproach::StreamingFromGeneral),
    //                   AccelType::KDTree),
    //  LightSamplerType::NoLightSampling, DirSamplerType::BSDF,
    //  TermProbType::MultiplierFunc, RngType::Sobel},

    // {IntersectionType(TAG(IntersectionApproach::MegaKernel),
    //                   AccelType::KDTree),
    //  LightSamplerType::RandomTriangle, DirSamplerType::BSDF,
    //  TermProbType::MultiplierFunc, RngType::Uniform},
    // {IntersectionType(TAG(IntersectionApproach::MegaKernel),
    //                   AccelType::KDTree),
    //  LightSamplerType::NoLightSampling, DirSamplerType::BSDF,
    //  TermProbType::MultiplierFunc, RngType::Uniform},
    // {IntersectionType(TAG(IntersectionApproach::MegaKernel),
    //                   AccelType::KDTree),
    //  LightSamplerType::RandomTriangle, DirSamplerType::Uniform,
    //  TermProbType::MultiplierFunc, RngType::Uniform},
    // {IntersectionType(TAG(IntersectionApproach::MegaKernel),
    //                   AccelType::KDTree),
    //  LightSamplerType::NoLightSampling, DirSamplerType::Uniform,
    //  TermProbType::MultiplierFunc, RngType::Uniform},
    // {IntersectionType(TAG(IntersectionApproach::StreamingFromGeneral),
    //                   AccelType::KDTree),
    //  LightSamplerType::RandomTriangle, DirSamplerType::BSDF,
    //  TermProbType::MultiplierFunc, RngType::Uniform},
    // {IntersectionType(TAG(IntersectionApproach::StreamingFromGeneral),
    //                   AccelType::KDTree),
    //  LightSamplerType::NoLightSampling, DirSamplerType::BSDF,
    //  TermProbType::MultiplierFunc, RngType::Uniform},
  }};
#else
  // build ALL possible values
  // Note that this SUBSTANTIALLY increases compile times
  constexpr static auto values = [] {
    constexpr auto tuple_values =
        AllValues<std::tuple<IntersectionType, LightSamplerType, DirSamplerType,
                             TermProbType, RngType>>;
    std::array<render::Settings::CompileTime, tuple_values.size()> out;
    std::transform(tuple_values.begin(), tuple_values.end(), out.begin(),
                   [](auto in) -> render::Settings::CompileTime {
                     auto [i, l, d, t, r] = in;

                     return {i, l, d, t, r};
                   });

    return out;
  }();
#endif
};

static_assert(AllValuesEnumerable<render::Settings::CompileTime>);

namespace render {
// outer lambda just to have scope
static_assert([] {
  constexpr auto values = AllValues<Settings::CompileTime>;

  // All values are unique
  static_assert([&] {
    for (unsigned i = 0; i < values.size(); ++i) {
      for (unsigned j = 0; j < i; ++j) {
        if (values[i] == values[j]) {
          return false;
        }
      }
    }

    return true;
  }());

  // default values are a valid dispatch value
  static_assert([&] {
    auto default_compile_time = Settings{}.compile_time();
    for (auto value : values) {
      if (value == default_compile_time) {
        return true;
      }
    }

    return false;
  }());

  return true;
}());
} // namespace render
