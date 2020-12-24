#pragma once

#include "lib/cuda/utils.h"

#include <vector>

// consider using construct method to override default initalization
template <typename T> struct UMAllocator {
  typedef T value_type;
  UMAllocator() {}
  template <typename U> UMAllocator(const UMAllocator<U> &other);

  T *allocate(size_t n);

  void deallocate(T *p, size_t n);
};

template <typename T> using ManangedMemVec = std::vector<T, UMAllocator<T>>;

template <typename T> T *UMAllocator<T>::allocate(size_t n) {
  T *ptr;
  if (n > 0) {
    CUDA_ERROR_CHK(cudaMallocManaged(&ptr, n * sizeof(T)));
  } else {
    ptr = NULL;
  }
  return ptr;
}

template <typename T> void UMAllocator<T>::deallocate(T *p, size_t) {
  CUDA_ERROR_CHK(cudaFree(p));
}
