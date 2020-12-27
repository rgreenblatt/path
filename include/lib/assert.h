#pragma once

#include <cassert> 

#define debug_assert(expr) assert(expr)

#define always_assert(expr)                                                    \
  (static_cast<bool>(expr)                                                     \
       ? void(0)                                                               \
       : __assert_fail(#expr, __FILE__, __LINE__, __ASSERT_FUNCTION))

#define unreachable_unchecked()                                                \
  debug_assert(false);                                                         \
  __builtin_unreachable();

#define unreachable() always_assert(false);
