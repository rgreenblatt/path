#pragma once

#include "intersect/bounded.h"
#include "intersect/intersectable.h"

namespace intersect {
template <typename T> concept Object = Bounded<T> &&Intersectable<T>;

struct MockObject : MockBounded, MockIntersectable {};
} // namespace intersect
