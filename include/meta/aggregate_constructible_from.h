#pragma once

template <typename T, typename... Args>
concept AggregateConstructibleFrom = requires(Args... args) {
  T{args...};
};
