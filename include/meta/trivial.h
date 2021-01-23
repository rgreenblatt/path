#pragma once

#include <type_traits>

template <typename T>
concept Trivial = std::is_trivially_copyable_v<T> &&
    std::is_trivially_default_constructible_v<T>;
