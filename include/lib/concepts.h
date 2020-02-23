#pragma once

#include <type_traits>
#include <array>

template <class T, template <typename...> class Template>
struct is_specialization : std::false_type {};

template <template <typename...> class Template, class... Args>
struct is_specialization<Template<Args...>, Template> : std::true_type {};

template <typename V, template <typename...> class Template>
concept IsSpecialization = is_specialization<std::decay_t<V>, Template>::value;

template <typename> struct is_std_array : std::false_type {};

template <typename T, std::size_t N>
struct is_std_array<std::array<T, N>> : std::true_type {};

template <typename V> concept IsStdArray = is_std_array<std::decay_t<V>>::value;