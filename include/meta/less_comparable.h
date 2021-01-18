#pragma once

#include <concepts>

// TODO: hana tuple compare work around...
template <typename T>
concept LessComparable = requires(const T &t) {
  { t < t } -> std::convertible_to<bool>;
};
