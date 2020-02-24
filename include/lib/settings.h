#pragma once

#include <concepts>

template <typename T> concept Setting = requires {
  std::equality_comparable<T>;
  std::default_initializable<T>;

  // TODO: serialization
};
