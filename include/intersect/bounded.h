#pragma once

#include "intersect/accel/aabb.h"
#include "meta/concepts.h"

#include <concepts>

namespace intersect {
template <typename T>
concept Bounded = requires(const T &t) {
  { t.bounds() } ->DecaysTo<accel::AABB>;
};

struct MockBounded {
  HOST_DEVICE accel::AABB bounds() const {
    return {Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero()};
  }
};

static_assert(Bounded<accel::AABB>);
} // namespace intersect
