#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <array>

namespace eigen_wrapper {
template <typename TIn> struct Wrapper {
  using T = TIn;

  static constexpr unsigned size = T::SizeAtCompileTime;
  std::array<typename T::Scalar, size> arr;

  constexpr Wrapper() = default;
  constexpr Wrapper(const T &value) {
    std::copy(value.data(), value.data() + size, arr.begin());
  }

  using MapT = Eigen::Map<T>;
  using CMapT = Eigen::Map<const T>;

  constexpr Wrapper(MapT value) : arr{value.arr(), value.arr() + size} {}
  constexpr Wrapper(CMapT value) : arr{value.arr(), value.arr() + size} {}

  constexpr operator CMapT() const { return CMapT(arr.data()); }
  constexpr operator MapT() { return MapT(arr.data()); }
  constexpr CMapT operator()() const { return CMapT(arr.data()); }
  constexpr MapT operator()() { return MapT(arr.data()); }

  constexpr auto data() { return arr.data(); }
  constexpr auto data() const { return arr.data(); }
  constexpr auto begin() { return arr.begin(); }
  constexpr auto begin() const { return arr.begin(); }
  constexpr auto end() { return arr.end(); }
  constexpr auto end() const { return arr.end(); }

  constexpr decltype(auto) operator[](int idx) { return (*this)()[idx]; }
  constexpr decltype(auto) operator[](int idx) const { return (*this)()[idx]; }

  constexpr decltype(auto) operator-() { return -(*this)(); }

#define EIGEN_WRAPPER_DECLARE_ASGN_OP(OP)                                      \
  template <typename Other>                                                    \
  constexpr decltype(auto) operator OP(const Other &r) {                       \
    constexpr bool is_same_t = std::same_as<Other, Wrapper>;                   \
    if constexpr (is_same_t) {                                                 \
      (*this)() OP r();                                                        \
    } else {                                                                   \
      (*this)() OP r;                                                          \
    }                                                                          \
    return *this;                                                              \
  }

  EIGEN_WRAPPER_DECLARE_ASGN_OP(+=)
  EIGEN_WRAPPER_DECLARE_ASGN_OP(-=)
  EIGEN_WRAPPER_DECLARE_ASGN_OP(*=)
  EIGEN_WRAPPER_DECLARE_ASGN_OP(/=)
};

#define EIGEN_WRAPPER_DECLARE_BIN_OP(OP)                                       \
  template <typename T, typename Other>                                        \
  constexpr decltype(auto) operator OP(const Wrapper<T> &l, const Other &r) {  \
    return l() OP r;                                                           \
  }                                                                            \
  template <typename Other, typename T>                                        \
  constexpr decltype(auto) operator OP(const Other &l, const Wrapper<T> &r) {  \
    return l OP r();                                                           \
  }

EIGEN_WRAPPER_DECLARE_BIN_OP(+)
EIGEN_WRAPPER_DECLARE_BIN_OP(-)
EIGEN_WRAPPER_DECLARE_BIN_OP(*)
EIGEN_WRAPPER_DECLARE_BIN_OP(/)
} // namespace eigen_wrapper
