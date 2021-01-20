#pragma once

#include "lib/info/printf_dbg.h"

#include <Eigen/Geometry>
#include <boost/hana/cartesian_product.hpp>
#include <boost/hana/ext/std/integer_sequence.hpp>
#include <boost/hana/unpack.hpp>

namespace printf_dbg {
namespace eigen {
namespace detail {
template <Formattable T, int rows, int cols> struct BaseEigenFmtImpl {
  static constexpr auto fmt =
      hana::unpack(std::make_index_sequence<rows>{}, [](auto... row) {
        return s("\n").join((
            void(row), s("{") +
                           boost::hana::unpack(std::make_index_sequence<cols>{},
                                               [&](auto... col) {
                                                 return s(", ").join(
                                                     (void(col), fmt_t<T>)...);
                                               }) +
                           s("}"))...);
      });

  template <typename V> static PRINTF_DBG_HOST_DEVICE auto vals(const V &val) {
    return hana::unpack(std::make_index_sequence<rows>{}, [&](auto... row) {
      return hana::flatten(hana::make_tuple([&](auto row) {
        return boost::hana::unpack(
            std::make_index_sequence<cols>{}, [&](auto... col) {
              return hana::make_tuple(val(row(), col())...);
            });
      }(row)...));
    });
  }
};
} // namespace detail
} // namespace eigen

template <typename Scalar, int rows, int cols>
struct FmtImpl<Eigen::Matrix<Scalar, rows, cols>>
    : eigen::detail::BaseEigenFmtImpl<Scalar, rows, cols> {};

template <typename Scalar, int rows, int cols>
struct FmtImpl<Eigen::Array<Scalar, rows, cols>>
    : eigen::detail::BaseEigenFmtImpl<Scalar, rows, cols> {};

template <typename Scalar, int dim, Eigen::TransformTraits type>
struct FmtImpl<Eigen::Transform<Scalar, dim, type>> {
  using MatrixType =
      decltype(std::declval<Eigen::Transform<Scalar, dim, type>>()
                   .matrix()
                   .eval());
  static constexpr auto fmt = fmt_t<MatrixType>();

  static PRINTF_DBG_HOST_DEVICE auto
  vals(const Eigen::Transform<Scalar, dim, type> &val) {
    fmt_vals(val.matrix());
  }
};
} // namespace printf_dbg
