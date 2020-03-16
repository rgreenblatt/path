#pragma once

#include "lib/cuda/utils.h"

// TODO: fix needing to include this here
#include <cereal-yaml/archives/yaml.hpp>

#include <concepts>

template <typename T>
concept Setting = requires(const T &data, cereal::YAMLInputArchive &archive) {
  std::semiregular<T>;

  {archive(data)};
};

struct EmptySettings {
  template <class Archive> void serialize(Archive &) {}

  HOST_DEVICE inline bool operator==(const EmptySettings &) const = default;
};

static_assert(Setting<EmptySettings>);
