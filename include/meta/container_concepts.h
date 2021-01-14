#pragma once

#include <concepts>
#include <type_traits>

template <typename T, typename... Args>
concept AggregateConstrucableFrom = requires(Args... args) {
  T{args...};
};

template <typename... T>
concept TriviallyDestructable = (... && std::is_trivially_destructible_v<T>);

template <typename... T>
concept Destructable = (... && std::is_destructible_v<T>);

template <typename... T>
concept TriviallyMoveConstructable =
    (... && std::is_trivially_move_constructible_v<T>);

template <typename... T>
concept MoveConstructable = (... && std::is_move_constructible_v<T>);

template <typename... T>
concept TriviallyCopyConstructable =
    (... && std::is_trivially_copy_constructible_v<T>);

template <typename... T>
concept CopyConstructable = (... && std::is_copy_constructible_v<T>);

template <typename... T>
concept TriviallyMoveAssignable = (... &&
                                   std::is_trivially_move_assignable_v<T>);

template <typename... T>
concept MoveAssignable = (... && std::is_move_assignable_v<T>);

template <typename... T>
concept TriviallyCopyAssignable = (... &&
                                   std::is_trivially_copy_assignable_v<T>);

template <typename... T>
concept CopyAssignable = (... && std::is_copy_assignable_v<T>);

template <typename... T>
concept TriviallyDefaultConstructable =
    (... && std::is_trivially_default_constructible_v<T>);

// can be copied by memcpy and similar
// https://en.cppreference.com/w/cpp/named_req/TriviallyCopyable
template <typename... T>
concept TriviallyCopyable = TriviallyCopyConstructable<T...> &&
    TriviallyMoveConstructable<T...> && TriviallyCopyAssignable<T...> &&
    TriviallyMoveAssignable<T...> && TriviallyDestructable<T...>;

template <typename... T>
concept TrivialType =
    TriviallyCopyable<T...> && TriviallyDefaultConstructable<T...>;
