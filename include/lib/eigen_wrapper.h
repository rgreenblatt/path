#pragma once

#include <Eigen/Geometry>

#include <algorithm>
#include <array>

// eigen wrapper which makes all constructors trivial
namespace eigen_wrapper {
template <typename TIn> struct Wrapper {
  using T = TIn;
  using Scalar = typename T::Scalar;

  static constexpr unsigned size = T::SizeAtCompileTime;
  std::array<Scalar, size> arr;

  constexpr Wrapper() = default;

  constexpr Wrapper(const T &value) {
    std::copy(value.data(), value.data() + size, arr.begin());
  }

  template <std::convertible_to<T> In>
  constexpr Wrapper(const In &value) : Wrapper(T(value)) {}

  using MapT = Eigen::Map<T>;
  using CMapT = Eigen::Map<const T>;

  constexpr Wrapper(MapT value) : arr{value.arr(), value.arr() + size} {}
  constexpr Wrapper(CMapT value) : arr{value.arr(), value.arr() + size} {}

  constexpr operator CMapT() const { return CMapT(arr.data()); }
  constexpr operator MapT() { return MapT(arr.data()); }
  constexpr CMapT operator()() const { return CMapT(arr.data()); }
  constexpr MapT operator()() { return MapT(arr.data()); }

  constexpr decltype(auto) sum() const { return (*this)().sum(); }
  constexpr decltype(auto) normalized() const { return (*this)().normalized(); }
  constexpr decltype(auto) norm() const { return (*this)().norm(); }
  constexpr decltype(auto) linear() const { return (*this)().linear(); }
  template <typename T> constexpr decltype(auto) dot(const T &other) const {
    return (*this)().dot(other);
  }
  template <typename T> constexpr decltype(auto) cross(const T &other) const {
    return (*this)().cross(other);
  }

  constexpr decltype(auto) matrix() const { return (*this)().matrix(); }
  constexpr decltype(auto) matrix() { return (*this)().matrix(); }
  constexpr decltype(auto) array() const { return (*this)().array(); }
  constexpr decltype(auto) array() { return (*this)().array(); }

  static constexpr auto Zero() { return T::Zero(); }
  static constexpr auto Constant(const Scalar &v) { return T::Constant(v); }

  constexpr decltype(auto) operator[](int idx) { return (*this)()[idx]; }
  constexpr decltype(auto) operator[](int idx) const { return (*this)()[idx]; }

  constexpr decltype(auto) x() { return (*this)().x(); }
  constexpr decltype(auto) y() { return (*this)().y(); }
  constexpr decltype(auto) z() { return (*this)().z(); }
  constexpr decltype(auto) w() { return (*this)().w(); }
  constexpr decltype(auto) x() const { return (*this)().x(); }
  constexpr decltype(auto) y() const { return (*this)().y(); }
  constexpr decltype(auto) z() const { return (*this)().z(); }
  constexpr decltype(auto) w() const { return (*this)().w(); }

  constexpr Wrapper eval() const { return *this; }

  template <typename T> constexpr decltype(auto) cast() const {
    return (*this)().template cast<T>();
  }

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
  }                                                                            \
  template <typename L, typename R>                                            \
  constexpr decltype(auto) operator OP(const Wrapper<L> &l,                    \
                                       const Wrapper<R> &r) {                  \
    return l() OP r();                                                         \
  }

EIGEN_WRAPPER_DECLARE_BIN_OP(+)
EIGEN_WRAPPER_DECLARE_BIN_OP(-)
EIGEN_WRAPPER_DECLARE_BIN_OP(*)
EIGEN_WRAPPER_DECLARE_BIN_OP(/)

template <typename Scalar, int rows, int cols>
requires(rows > 0 &&
         cols > 0) using Array = Wrapper<Eigen::Array<Scalar, rows, cols>>;
template <typename Scalar, int rows, int cols>
requires(rows > 0 &&
         cols > 0) using Matrix = Wrapper<Eigen::Matrix<Scalar, rows, cols>>;
template <typename Scalar, int dim, int mode>
requires(dim >
         0) using Transform = Wrapper<Eigen::Transform<Scalar, dim, mode>>;

template <typename Scalar, int rows> using Vector = Matrix<Scalar, rows, 1>;
template <typename Scalar, int cols> using RowVector = Matrix<Scalar, 1, cols>;

template <typename Scalar> using Array1 = Array<Scalar, 1, 1>;
template <typename Scalar> using Array2 = Array<Scalar, 2, 1>;
template <typename Scalar> using Array3 = Array<Scalar, 3, 1>;
template <typename Scalar> using Array4 = Array<Scalar, 4, 1>;
template <typename Scalar> using Matrix1 = Matrix<Scalar, 1, 1>;
template <typename Scalar> using Matrix2 = Matrix<Scalar, 2, 2>;
template <typename Scalar> using Matrix3 = Matrix<Scalar, 3, 3>;
template <typename Scalar> using Matrix4 = Matrix<Scalar, 4, 4>;
template <typename Scalar> using Vector1 = Vector<Scalar, 1>;
template <typename Scalar> using Vector2 = Vector<Scalar, 2>;
template <typename Scalar> using Vector3 = Vector<Scalar, 3>;
template <typename Scalar> using Vector4 = Vector<Scalar, 4>;
template <typename Scalar> using RowVector1 = RowVector<Scalar, 1>;
template <typename Scalar> using RowVector2 = RowVector<Scalar, 2>;
template <typename Scalar> using RowVector3 = RowVector<Scalar, 3>;
template <typename Scalar> using RowVector4 = RowVector<Scalar, 4>;
template <typename Scalar> using Affine2 = Transform<Scalar, 2, Eigen::Affine>;
template <typename Scalar> using Affine3 = Transform<Scalar, 3, Eigen::Affine>;

template <typename Scalar> using RowVector4 = RowVector<Scalar, 4>;

using Array1f = Array1<float>;
using Array2f = Array2<float>;
using Array3f = Array3<float>;
using Array4f = Array4<float>;
using Matrix1f = Matrix1<float>;
using Matrix2f = Matrix2<float>;
using Matrix3f = Matrix3<float>;
using Matrix4f = Matrix4<float>;
using Vector1f = Vector1<float>;
using Vector2f = Vector2<float>;
using Vector3f = Vector3<float>;
using Vector4f = Vector4<float>;
using RowVector1f = RowVector1<float>;
using RowVector2f = RowVector2<float>;
using RowVector3f = RowVector3<float>;
using RowVector4f = RowVector4<float>;
using Affine2f = Affine2<float>;
using Affine3f = Affine3<float>;
using Array1d = Array1<double>;
using Array2d = Array2<double>;
using Array3d = Array3<double>;
using Array4d = Array4<double>;
using Matrix1d = Matrix1<double>;
using Matrix2d = Matrix2<double>;
using Matrix3d = Matrix3<double>;
using Matrix4d = Matrix4<double>;
using Vector1d = Vector1<double>;
using Vector2d = Vector2<double>;
using Vector3d = Vector3<double>;
using Vector4d = Vector4<double>;
using RowVector1d = RowVector1<double>;
using RowVector2d = RowVector2<double>;
using RowVector3d = RowVector3<double>;
using RowVector4d = RowVector4<double>;
using Affine2d = Affine2<double>;
using Affine3d = Affine3<double>;
} // namespace eigen_wrapper
