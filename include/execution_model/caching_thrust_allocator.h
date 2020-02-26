#pragma once

#include "execution_model/execution_model.h"
#include "lib/cuda/utils.h"

#include <thrust/host_vector.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/cuda/vector.h>

#include <iostream>
#include <map>

// thrust-cuda-tip-reuse-temporary-buffers-across-transforms
template <ExecutionModel execution_model> class CachingThrustAllocator {
public:
  // just allocate bytes
  typedef char value_type;

  CachingThrustAllocator() {}

  CachingThrustAllocator(const CachingThrustAllocator &) = delete;

  CachingThrustAllocator(CachingThrustAllocator &&) = default;

  ~CachingThrustAllocator() {
    // free all allocations when cached_allocator goes out of scope
    free_all();
  }

  char *allocate(std::ptrdiff_t num_bytes) {
    char *result = 0;

    // search the cache for a free block
    free_blocks_type::iterator free_block = free_blocks.find(num_bytes);

    if (free_block != free_blocks.end()) {
      // get the pointer
      result = free_block->second;

      // erase from the free_blocks map
      free_blocks.erase(free_block);
    } else {
      // no allocation of the right size exists
      // create a new one with cuda::malloc
      // throw if cuda::malloc can't satisfy the request
      try {
        // allocate memory and convert cuda::pointer to raw pointer
        result = malloc_impl(num_bytes);
      } catch (std::runtime_error &e) {
        throw;
      }
    }

    // insert the allocated pointer into the allocated_blocks map
    allocated_blocks.insert(std::make_pair(result, num_bytes));

    return result;
  }

  void deallocate(char *ptr, size_t /* n */) {
    // erase the allocated block from the allocated blocks map
    allocated_blocks_type::iterator iter = allocated_blocks.find(ptr);
    std::ptrdiff_t num_bytes = iter->second;
    allocated_blocks.erase(iter);

    // insert the block into the free blocks map
    free_blocks.insert(std::make_pair(num_bytes, ptr));
  }

private:
  typedef std::multimap<std::ptrdiff_t, char *> free_blocks_type;
  typedef std::map<char *, std::ptrdiff_t> allocated_blocks_type;

  free_blocks_type free_blocks;
  allocated_blocks_type allocated_blocks;

  void free_all() {
    // deallocate all outstanding blocks in both lists
    for (free_blocks_type::iterator i = free_blocks.begin();
         i != free_blocks.end(); i++) {
      // transform the pointer to cuda::pointer before calling cuda::free
      free_impl(i->second);
    }

    for (allocated_blocks_type::iterator i = allocated_blocks.begin();
         i != allocated_blocks.end(); i++) {
      // transform the pointer to cuda::pointer before calling cuda::free
      free_impl(i->first);
    }
  }

  char *malloc_impl(std::ptrdiff_t num_bytes) {
    if constexpr (execution_model == ExecutionModel::GPU) {
      return thrust::cuda::malloc<char>(num_bytes).get();
    } else {
      return reinterpret_cast<char *>(malloc(num_bytes));
    }
  }

  void free_impl(char *ptr) {
    if constexpr (execution_model == ExecutionModel::GPU) {
      thrust::cuda::free(thrust::cuda::pointer<char>(ptr));
    } else {
      free(ptr);
    }
  }
};
