#pragma once

#include "intersect/accel/aabb.h"
#include "meta/decays_to.h"
#include "meta/mock.h"

namespace intersect {
template <typename T> concept Bounded = requires(const T &t) {
  { t.bounds() }
  ->DecaysTo<accel::AABB>;
};

struct MockBounded : MockNoRequirements {
  HOST_DEVICE accel::AABB bounds() const;
};

static_assert(Bounded<MockBounded>);
static_assert(Bounded<accel::AABB>);
} // namespace intersect
