#pragma once

#include "data_structure/detail/vector_like.h"
#include "lib/span.h"
#include "meta/concepts.h"

template <typename From, detail::VectorLike VecTo>
void copy_to_vec(const From &f, VecTo &t) {
  t.resize(f.size());
  thrust::copy(f.begin(), f.end(), t.begin());
}

template <typename From, typename To>
concept CopyableToVec = requires(const From &f, To &t) {
  copy_to_vec(f, t);
};
