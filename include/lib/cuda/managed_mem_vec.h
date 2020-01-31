#pragma once

#include "lib/cuda/utils.h"

#include <vector>

template <class T> struct UMAllocator {
  typedef T value_type;
  UMAllocator() {}
  template <class U> UMAllocator(const UMAllocator<U> &other);

  T *allocate(size_t n);

  void deallocate(T *p, size_t n);
};

template <class T> using ManangedMemVec = std::vector<T, UMAllocator<T>>;

template <class T> T *UMAllocator<T>::allocate(size_t n) {
  T *ptr;
  if (n > 0) {
    CUDA_ERROR_CHK(cudaMallocManaged(&ptr, n * sizeof(T)));
  } else {
    ptr = NULL;
  }
  return ptr;
}

template <class T> void UMAllocator<T>::deallocate(T *p, size_t) {
  CUDA_ERROR_CHK(cudaFree(p));
}
