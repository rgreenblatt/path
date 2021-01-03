#pragma once

#include "lib/attribute.h"
#include "lib/serialize.h"
#include "meta/mock.h"

#include <concepts>
#include <type_traits>
#include <utility>

// This is the concept of a serializable setting (using cereal)

// To avoid having to include cereal here we mock the archive object
struct MockArchive : MockNoRequirements {};

template <typename T>
concept Setting = requires(T &data, MockArchive &archive) {
  requires std::regular<T>;

  data.serialize(archive);
};

struct EmptySettings {
  template <typename Archive> void serialize(Archive &) {}

  ATTR_PURE constexpr bool operator==(const EmptySettings &) const = default;
};

static_assert(Setting<EmptySettings>);
