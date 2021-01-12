#pragma once

#include "data_structure/get_ptr.h"
#include "data_structure/get_size.h"

#include <concepts>

template <typename Vec>
concept Vector = requires(Vec &v, unsigned n, typename Vec::value_type x) {
  requires std::movable<Vec>;
  typename Vec::value_type;
  v.resize(n);
  v.resize(n, x);
  requires GetPtrForElem<const Vec &, const typename Vec::value_type>;
  requires GetPtrForElem<Vec &, typename Vec::value_type>;
  requires GetSize<Vec>;
};
