#pragma once

#include "lib/settings.h"

namespace render {
// TODO: add more
enum class TermProbType { DirectLightingOnly, Constant, MultiplierFunc };

template <TermProbType type> struct TermProbSettings;

template <>
struct TermProbSettings<TermProbType::DirectLightingOnly> : EmptySettings {};

template <> struct TermProbSettings<TermProbType::Constant> {
  float prob = 0.5f;

  template <class Archive> void serialize(Archive &archive) {
    archive(CEREAL_NVP(prob));
  }
};

template <> struct TermProbSettings<TermProbType::MultiplierFunc> {
  float exp = 10.0f;
  float min_prob = 0.05f;

  template <class Archive> void serialize(Archive &archive) {
    archive(CEREAL_NVP(exp), CEREAL_NVP(min_prob));
  }
};

static_assert(Setting<TermProbSettings<TermProbType::DirectLightingOnly>>);
static_assert(Setting<TermProbSettings<TermProbType::Constant>>);
static_assert(Setting<TermProbSettings<TermProbType::MultiplierFunc>>);
}; // namespace render
