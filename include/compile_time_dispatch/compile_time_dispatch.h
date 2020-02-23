#pragma once

#include <array>

// trait:
// count: constexpr unsigned
// values: constexpr array
template <typename T, typename = void> struct CompileTimeDispatchable;
