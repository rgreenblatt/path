#pragma once

#include "meta/mock.h"

#include <cereal-yaml/archives/yaml.hpp>

#include <concepts>

// This is the concept of a serializable setting (using cereal)

// To avoid having to include cereal here we mock the archive object
struct MockArchive : MockNoRequirements {};

template <typename T>
concept Setting = requires(T &data, MockArchive &archive) {
  requires std::regular<T>;

  data.serialize(archive);
};

struct EmptySettings {
  template <class Archive> void serialize(Archive &) {}

  constexpr bool operator==(const EmptySettings &) const = default;
};

static_assert(Setting<EmptySettings>);

// convenience macro for use in serialize
#define NVP(T) CEREAL_NVP(T)
