#pragma once

template <typename T, typename... Args>
concept AggregateConstrucableFrom = requires(Args... args) {
  T{args...};
};
