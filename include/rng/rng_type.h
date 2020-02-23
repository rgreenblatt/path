#pragma once

/* #include "lib/trait.h" */

#include <concepts>

namespace rng {
enum class RngType { Uniform, Halton, Sobel };

template <typename Impl>
concept RngTrait = requires{
  typename Impl::Settings;
  std::semiregular<typename Impl::Settings>;
};

} // namespace rng
