#pragma once

#include <array>
#include <concepts>
#include <type_traits>

// special case of SpecializationOf required for array because of size param
template <typename> struct is_std_array : std::false_type {};

template <typename T, std::size_t N>
struct is_std_array<std::array<T, N>> : std::true_type {};

template <typename V>
concept StdArraySpecialization = is_std_array<std::decay_t<V>>::value;

template <typename V, typename T> concept StdArrayOfType = requires {
  requires is_std_array<std::decay_t<V>>::value;
  typename std::decay_t<V>::value_type;
  requires std::same_as<typename std::decay_t<V>::value_type, T>;
};
