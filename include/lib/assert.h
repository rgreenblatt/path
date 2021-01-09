#pragma once

#include <cassert>

extern "C" {
// always forward declare __assert_fail regardless of NDEBUG
extern void __assert_fail(const char *__assertion, const char *__file,
                          unsigned int __line, const char *__function) __THROW
    __attribute__((__noreturn__));
}

// assert which is disabled by NDEBUG
#define debug_assert(expr) assert(expr)

// debug assert which allows the compile to assume the statement is always true
// when NDEBUG is enabled
#ifdef NDEBUG
#define debug_assert_assume(expr) __builtin_assume(expr);
#else
#define debug_assert_assume(expr) debug_assert(expr)
#endif

// assert which is never disabled
#define always_assert(expr)                                                    \
  (static_cast<bool>(expr) ? void(0)                                           \
                           : __assert_fail(#expr, __FILE__, __LINE__,          \
                                           __extension__ __PRETTY_FUNCTION__))

// assert that a section is unreachable or if NDEBUG is defined
// don't assert, cause UB if section is reached
#define unreachable_unchecked()                                                \
  debug_assert(false);                                                         \
  __builtin_unreachable();

// assert that section is unreachable (always)
#define unreachable() always_assert(false);

inline constexpr bool debug_build =
#ifdef NDEBUG
    false
#else
    true
#endif
    ;