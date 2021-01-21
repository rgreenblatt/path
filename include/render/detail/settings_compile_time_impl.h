#pragma once

#include "meta/all_values/impl/as_tuple.h"
#include "meta/all_values/impl/tuple.h"
#include "meta/as_tuple/as_tuple.h"
#include "meta/unpack_to.h"
#include "render/settings.h"

// compile times don't change much from small constant values to 1...
// compile times do substantially increase for large number of possibilities
// #define FORCE_BUILD_ALL
#if !defined(BUILD_ALL) && !defined(FORCE_BUILD_ALL)
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
  constexpr static std::array<render::Settings::CompileTime, 2> values = {{
      {IntersectionType(TagV<IntersectionApproach::MegaKernel>,
                        AccelType::KDTree),
       LightSamplerType::RandomTriangle, DirSamplerType::BSDF,
       TermProbType::MultiplierFunc, RngType::Sobel},
      // {IntersectionType(TagV<IntersectionApproach::MegaKernel>,
      //                   AccelType::KDTree),
      //  LightSamplerType::NoLightSampling, DirSamplerType::BSDF,
      //  TermProbType::MultiplierFunc, RngType::Sobel},
      // {IntersectionType(TagV<IntersectionApproach::MegaKernel>,
      //                   AccelType::KDTree),
      //  LightSamplerType::RandomTriangle, DirSamplerType::Uniform,
      //  TermProbType::MultiplierFunc, RngType::Sobel},
      // {IntersectionType(TagV<IntersectionApproach::MegaKernel>,
      //                   AccelType::KDTree),
      //  LightSamplerType::NoLightSampling, DirSamplerType::Uniform,
      //  TermProbType::MultiplierFunc, RngType::Sobel},
      // {IntersectionType(TagV<IntersectionApproach::StreamingFromGeneral>,
      //                   AccelType::KDTree),
      //  LightSamplerType::RandomTriangle, DirSamplerType::BSDF,
      //  TermProbType::MultiplierFunc, RngType::Sobel},
      // {IntersectionType(TagV<IntersectionApproach::StreamingFromGeneral>,
      //                   AccelType::KDTree),
      //  LightSamplerType::NoLightSampling, DirSamplerType::BSDF,
      //  TermProbType::MultiplierFunc, RngType::Sobel},

      {IntersectionType(TagV<IntersectionApproach::MegaKernel>,
                        AccelType::KDTree),
       LightSamplerType::RandomTriangle, DirSamplerType::BSDF,
       TermProbType::MultiplierFunc, RngType::Uniform},
      // {IntersectionType(TagV<IntersectionApproach::MegaKernel>,
      //                   AccelType::KDTree),
      //  LightSamplerType::NoLightSampling, DirSamplerType::BSDF,
      //  TermProbType::MultiplierFunc, RngType::Uniform},
      // {IntersectionType(TagV<IntersectionApproach::MegaKernel>,
      //                   AccelType::KDTree),
      //  LightSamplerType::RandomTriangle, DirSamplerType::Uniform,
      //  TermProbType::MultiplierFunc, RngType::Uniform},
      // {IntersectionType(TagV<IntersectionApproach::MegaKernel>,
      //                   AccelType::KDTree),
      //  LightSamplerType::NoLightSampling, DirSamplerType::Uniform,
      //  TermProbType::MultiplierFunc, RngType::Uniform},
      // {IntersectionType(TagV<IntersectionApproach::StreamingFromGeneral>,
      //                   AccelType::KDTree),
      //  LightSamplerType::RandomTriangle, DirSamplerType::BSDF,
      //  TermProbType::MultiplierFunc, RngType::Uniform},
      // {IntersectionType(TagV<IntersectionApproach::StreamingFromGeneral>,
      //                   AccelType::KDTree),
      //  LightSamplerType::NoLightSampling, DirSamplerType::BSDF,
      //  TermProbType::MultiplierFunc, RngType::Uniform},
  }};
};
#else
// build ALL possible values
// Note that this SUBSTANTIALLY increases compile times
// If code isn't generated (e.g.) -fsyntax-only, this is actually ok
// (but still increases build times...)
template <> struct AsTupleImpl<render::Settings::CompileTime> {
  constexpr auto static as_tuple(const render::Settings::CompileTime &v) {
    return make_meta_tuple(v.intersection_type, v.light_sampler_type,
                           v.dir_sampler_type, v.term_prob_type, v.rng_type);
  }

  template <typename T> constexpr auto static from_tuple(const T &v) {
    return unpack_to<render::Settings::CompileTime>(v);
  }
};

static_assert(AsTuple<render::Settings::CompileTime>);
#endif

static_assert(AllValuesEnumerable<render::Settings::CompileTime>);

// default settings value is included
static_assert([] {
  auto default_value = render::Settings{}.compile_time();
  for (auto value : AllValues<render::Settings::CompileTime>) {
    if (value == default_value) {
      return true;
    }
  }
  return false;
}());
