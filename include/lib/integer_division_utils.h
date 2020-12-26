#pragma once

constexpr unsigned ceil_divide(unsigned l, unsigned r) {
  return (l + r - 1) / r;
};

constexpr unsigned make_divisable(unsigned v, unsigned divisable_by) {
  return ceil_divide(v, divisable_by) * divisable_by;
}
