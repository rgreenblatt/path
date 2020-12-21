#pragma once

#include "data_structure/vector.h"
#include "lib/cuda/managed_mem_vec.h"

template <typename T> using HostDeviceVector = ManangedMemVec<T>;
static_assert(Vector<HostDeviceVector<int>>);
