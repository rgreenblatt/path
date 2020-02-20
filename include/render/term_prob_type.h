#pragma once

namespace render {
// TODO: add more
enum class TermProbType { Uniform, MultiplierNorm };

template <TermProbType type> struct TermProbSettings;

template <> struct TermProbSettings<TermProbType::Uniform> { float prob; };

template <> struct TermProbSettings<TermProbType::MultiplierNorm> {
  // -2 to 2 (continuous)
  // -2 is most concave, 0 is linear, and 2 is most convex
  float convexity_scale;
};
}; // namespace render
