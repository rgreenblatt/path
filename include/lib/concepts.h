#pragma once

#include <array>
#include <type_traits>

template <class T, template <typename...> class Template>
struct is_specialization : std::false_type {};

template <template <typename...> class Template, class... Args>
struct is_specialization<Template<Args...>, Template> : std::true_type {};

template <typename V, template <typename...> class Template>
concept SpecializationOf = is_specialization<std::decay_t<V>, Template>::value;

// special case array because of size param
template <typename> struct is_std_array : std::false_type {};

template <typename T, std::size_t N>
struct is_std_array<std::array<T, N>> : std::true_type {};

template <typename V>
concept StdArraySpecialization = is_std_array<std::decay_t<V>>::value;

template<typename...> concept TrueConcept = true;
