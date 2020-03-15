#pragma once

#include <concepts>

#include <yaml-cpp/yaml.h>  // IWYU pragma: keep
#include <cereal-yaml/archives/yaml.hpp>

template <typename T>
concept Setting = requires(const T &data, cereal::YAMLInputArchive &archive) {
  std::semiregular<T>;

  { archive(data) };
};

struct EmptySettings {
  template <class Archive> void serialize(Archive &) {}
  
  HOST_DEVICE inline bool
  operator==(const EmptySettings &) const = default;
};

static_assert(Setting<EmptySettings>);
