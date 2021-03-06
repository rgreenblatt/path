#pragma once

#include <type_traits>

template <typename T, template <typename...> class Template>
struct is_specialization : std::false_type {};

template <template <typename...> class Template, class... Args>
struct is_specialization<Template<Args...>, Template> : std::true_type {};

template <typename V, template <typename...> class Template>
concept ExactSpecializationOf = is_specialization<V, Template>::value;

template <typename V, template <typename...> class Template>
concept SpecializationOf = ExactSpecializationOf<std::decay_t<V>, Template>;
