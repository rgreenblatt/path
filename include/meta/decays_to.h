#pragma once

#include <concepts>
#include <type_traits>

template <typename From, typename To>
concept DecaysTo = std::same_as<std::decay_t<From>, To>;
