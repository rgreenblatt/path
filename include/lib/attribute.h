#pragma once

// Doesn't have to actually be pure, but value shouldn't be discarded
#define ATTR_NO_DISCARD_PURE [[nodiscard("pure func")]]
#define ATTR_PURE [[gnu::pure]]

// attribute for function which are pure except for debug_assert -
// they are pure if NDEBUG is define and the compiler should
// always issue a warning if the return value is unused...
#ifdef NDEBUG
#define ATTR_PURE_NDEBUG ATTR_PURE
#else
#define ATTR_PURE_NDEBUG ATTR_NO_DISCARD_PURE
#endif
