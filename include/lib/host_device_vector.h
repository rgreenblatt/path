#pragma once

#include "lib/cuda/managed_mem_vec.h"

template <typename T> using HostDeviceVector = ManangedMemVec<T>;
