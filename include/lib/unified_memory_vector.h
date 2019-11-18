#pragma once

#include <cuda.h>

#include <cstdio>
#include <vector>

template <class T> struct UMAllocator {
  typedef T value_type;
  UMAllocator() {}
  template <class U> UMAllocator(const UMAllocator<U> &other);

  T *allocate(size_t n);

  void deallocate(T *p, size_t n);
};

template <class T> using AllocVec = std::vector<T, UMAllocator<T>>;

template <class T> T *UMAllocator<T>::allocate(size_t n) {
  T *ptr;
  if (n > 0) {
    cudaMallocManaged(&ptr, n * sizeof(T));
  } else {
    ptr = NULL;
  }
  return ptr;
}

template <class T> void UMAllocator<T>::deallocate(T *p, size_t) {
  cudaFree(p);
}
