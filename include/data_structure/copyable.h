#include "data_structure/detail/vector_like.h"

template <detail::VectorLike VecFrom, detail::VectorLike VecTo>
requires std::same_as<typename VecFrom::value_type, typename VecTo::value_type>
void copy_to(const VecFrom &f, VecTo &t) {
  t.resize(f.size());
  thrust::copy(f.begin(), f.end(), t.begin());
}

template <typename VecFrom, typename VecTo>
concept Copyable = requires(const VecFrom &f, VecTo &t) {
  copy_to(f, t);
};
