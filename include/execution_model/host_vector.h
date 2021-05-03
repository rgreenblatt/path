#pragma once

#include "data_structure/vector.h"
#include "lib/vector_type.h"

template <typename T> using HostVector = VectorT<T>;

static_assert(Vector<HostVector<int>>);
