#pragma once

#include <cassert>

// assert which is disabled by NDEBUG
#define debug_assert(expr) assert(expr)

// assert which is never disabled
#define always_assert(expr)                                                    \
  (static_cast<bool>(expr)                                                     \
       ? void(0)                                                               \
       : __assert_fail(#expr, __FILE__, __LINE__, __ASSERT_FUNCTION))

// assert that a section is unreachable or if NDEBUG is defined
// don't assert, cause UB if section is reached
#define unreachable_unchecked()                                                \
  debug_assert(false);                                                         \
  __builtin_unreachable();

// assert that section is unreachable (always)
#define unreachable() always_assert(false);
