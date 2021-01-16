#pragma once

#include "meta/all_values_tuple.h"
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
  // #define FORCE_BUILD_ALL

#if !defined(BUILD_ALL) && !defined(FORCE_BUILD_ALL)
  constexpr static std::array<render::Settings::CompileTime, 2> values = {
      { {IntersectionType(TagV<IntersectionApproach::MegaKernel>,
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
#else
  // build ALL possible values
  // Note that this SUBSTANTIALLY increases compile times
  constexpr static auto values = [] {
    constexpr auto tuple_values =
        AllValues<MetaTuple<IntersectionType, LightSamplerType, DirSamplerType,
                            TermProbType, RngType>>;
    std::array<render::Settings::CompileTime, tuple_values.size()> out;
    std::transform(tuple_values.begin(), tuple_values.end(), out.begin(),
                   [](auto in) {
                     return boost::hana::unpack(
                         in, [](auto &&...v) -> render::Settings::CompileTime {
                           return {v...};
                         });
                   });

    return out;
  }();
#endif
};

static_assert(AllValuesEnumerable<render::Settings::CompileTime>);
