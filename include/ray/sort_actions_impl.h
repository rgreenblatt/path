#pragma once

#include "lib/cuda_utils.h"
#include "ray/sort_actions.h"
#include "ray/ray_utils.h"

namespace ray {
namespace detail {
inline HOST_DEVICE void sort_traversal_actions(const Traversal &traversal,
                                               Span<Action> actions) {
  unsigned i = traversal.start;
  unsigned j = traversal.start + 1;

  if (traversal.end <= traversal.start + 1) {
    return;
  }

  // insertion sort
  while (true) {
    bool is_greater = actions[j - 1].min_dist > actions[j].min_dist;
    if (is_greater) {
      swap(actions[j - 1], actions[j]);
      j--;
    }

    if (!is_greater || j == traversal.start) {
      i++;
      j = i;
      if (i == traversal.end) {
        break;
      }
    }
  }

#ifndef NDEBUG
  float dist = std::numeric_limits<float>::lowest();
  for (unsigned i = traversal.start; i < traversal.end; i++) {
    assert(actions[i].min_dist >= dist);
    dist = actions[i].min_dist;
  }
#endif
}
} // namespace detail
} // namespace ray
