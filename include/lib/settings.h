#pragma once

// TODO: fix needing to include this here (should be possible to have less?)
#include <cereal-yaml/archives/yaml.hpp>

#include <concepts>

template <typename T>
concept Setting = requires(const T &data, cereal::YAMLInputArchive &archive) {
  std::regular<T>;

  {archive(data)};
};

struct EmptySettings {
  template <class Archive> void serialize(Archive &) {}

  constexpr inline bool operator==(const EmptySettings &) const = default;
};

static_assert(Setting<EmptySettings>);
