#pragma once

#include "data_structure/detail/vector_like.h"

#include <thrust/copy.h>

template <typename From, detail::VectorLike VecTo>
void copy_to_vec(const From &f, VecTo &t) {
  t.resize(f.size());
  thrust::copy(f.begin(), f.end(), t.begin());
}

// this concept isn't very strict with the From type...
template <typename From, typename To>
concept CopyableToVec = requires(const From &f, To &t) {
  f.size();
  f.begin();
  f.end();
  detail::VectorLike<To>;
};
