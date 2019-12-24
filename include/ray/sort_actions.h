#pragma once

#include "lib/span.h"
#include "ray/traversal_grid.h"

namespace ray {
namespace detail {
void sort_actions(Span<const Traversal, false> traversals, Span<Action> actions);
void sort_actions_cpu(Span<const Traversal, false> traversals, Span<Action> actions);
} // namespace detail
} // namespace ray
