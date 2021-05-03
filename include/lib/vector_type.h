#pragma once

#ifdef NDEBUG
#include <vector>
#else
#include <debug/vector>
#endif

// Thine shalt never use a vector in debug without bounds checking.
// So sayeth the code!
template <typename T, typename Alloc = std::allocator<T>>
using VectorT =
#ifdef NDEBUG
    std::vector<T, Alloc>
#else
    __gnu_debug::vector<T, Alloc>
#endif
    ;
