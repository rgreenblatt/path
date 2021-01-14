#pragma once

#include <array>
#include <concepts>
#include <type_traits>

namespace std_array_specialization {
namespace detail {
// special case of SpecializationOf required for array because of size param
template <typename> struct IsStdArrayImpl : std::false_type {};

template <typename T, std::size_t N>
struct IsStdArrayImpl<std::array<T, N>> : std::true_type {};
} // namespace detail
} // namespace std_array_specialization

template <typename V>
concept StdArraySpecialization =
    std_array_specialization::detail::IsStdArrayImpl<std::decay_t<V>>::value;

template <typename V, typename T>
concept StdArrayOfType = requires {
  requires std_array_specialization::detail::IsStdArrayImpl<
      std::decay_t<V>>::value;
  typename std::decay_t<V>::value_type;
  requires std::same_as<typename std::decay_t<V>::value_type, T>;
};
