#pragma once

#include "data_structure/vector.h"

#include <vector>

template <typename T> using HostVector = std::vector<T>;
static_assert(Vector<HostVector<int>>);
