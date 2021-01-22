#pragma once

#include "lib/array_transform.h"
#include "meta/all_values/all_values.h"
#include "meta/all_values/impl/tuple.h"
#include "meta/as_tuple/as_tuple.h"
#include "meta/tuple.h"

template <AsTuple T> struct AllValuesImpl<T> {
  static constexpr auto values =
      array_transform(AllValues<AsTupleT<T>>,
                      [](auto in) { return AsTupleImpl<T>::from_tuple(in); });
};
