#pragma once

namespace rng {
enum class RngType { Uniform, Halton, Sobel };

template <RngType type> struct RngSettings;

template <> struct RngSettings<RngType::Uniform> {};

template <> struct RngSettings<RngType::Halton> {};

template <> struct RngSettings<RngType::Sobel> {};
} // namespace rng
