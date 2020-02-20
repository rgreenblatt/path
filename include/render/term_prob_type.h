#pragma once

namespace render {
// TODO: add more
enum class TermProbType { Uniform, MultiplierNorm };

template <TermProbType type> struct TermProbSettings;

template <> struct TermProbSettings<TermProbType::Uniform> { float prob; };

template <> struct TermProbSettings<TermProbType::MultiplierNorm> {
  // TODO: settings for some function of multiplier...
};
}; // namespace render
