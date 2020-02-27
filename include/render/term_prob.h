#pragma once

namespace render {
// TODO: add more
enum class TermProbType { DirectLightingOnly, Constant, MultiplierFunc };

template <TermProbType type> struct TermProbSettings;

template <> struct TermProbSettings<TermProbType::DirectLightingOnly> {};

template <> struct TermProbSettings<TermProbType::Constant> {
  float prob = 0.5f;
};

template <> struct TermProbSettings<TermProbType::MultiplierFunc> {
  float exp = 10.0f;
  float min_prob = 0.05f;
};
}; // namespace render
