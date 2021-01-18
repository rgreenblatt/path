#pragma once

#include "meta/enum.h"

#include <iostream>
#include <string>

namespace cereal {
template <typename Archive, Enum T>
inline std::string save_minimal(const Archive &, const T &enum_v) {
  return std::string(magic_enum::enum_name(enum_v));
}

template <typename Archive, Enum T>
inline void load_minimal(const Archive &, T &enum_v, const std::string &s) {
  auto val_op = magic_enum::enum_cast<T>(s);
  if (val_op.has_value()) {
    enum_v = val_op.value();
  } else {
    std::cerr << "failed to load enum with string: " << s << std::endl;
    abort();
  }
}
} // namespace cereal
