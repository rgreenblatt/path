#pragma once

#include "lib/settings.h"

namespace render {
// TODO: add more
enum class TermProbType { Constant, NIters, MultiplierFunc };

template <TermProbType type> struct TermProbSettings;

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

template <> struct TermProbSettings<TermProbType::NIters> {
  unsigned iters = 1;

  template <class Archive> void serialize(Archive &archive) {
    archive(CEREAL_NVP(iters));
  }
};

static_assert(Setting<TermProbSettings<TermProbType::Constant>>);
static_assert(Setting<TermProbSettings<TermProbType::MultiplierFunc>>);
static_assert(Setting<TermProbSettings<TermProbType::NIters>>);
}; // namespace render
