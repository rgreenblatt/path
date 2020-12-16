#pragma once

#include "lib/cuda/managed_mem_vec.h"
#include "data_structure/vector.h"

template <typename T> using HostDeviceVector = ManangedMemVec<T>;
static_assert(Vector<HostDeviceVector<int>>);
