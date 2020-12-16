#pragma once

#include "data_structure/get_ptr.h"
#include "data_structure/get_size.h"

#include <concepts>

template <typename Vec>
concept Vector = requires(Vec &v, typename Vec::value_type x) {
  std::movable<Vec>;
  typename Vec::value_type;
  v.resize(unsigned());
  v.resize(unsigned(), x);
  typename GetPtrT<Vec, typename Vec::value_type>;
  GetPtr<GetPtrT<Vec, typename Vec::value_type>, Vec>;
  typename GetSizeT<Vec>;
  GetSize<Vec>;
};
