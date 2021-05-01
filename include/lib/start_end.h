#pragma once

template <typename T> struct StartEnd {
  T start;
  T end;

  constexpr T size() const { return end - start; }
  constexpr bool empty() const { return start == end; }

  constexpr bool operator==(const StartEnd &other) const = default;
  constexpr auto operator<=>(const StartEnd &other) const = default;
};
