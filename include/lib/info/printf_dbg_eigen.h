#pragma once

#include "lib/info/printf_dbg.h"

#include <Eigen/Geometry>

namespace printf_dbg {
namespace eigen {
namespace detail {
template <int rows, int cols, typename T>
PRINTF_DBG_HOST_DEVICE auto eigen_fmt_impl(const T &val) {
  return boost::hana::unpack(
      std::make_index_sequence<rows>{}, [&](auto... row) {
        return join(s_str("\n"), [&](auto row) {
          auto out =
              s(s_str("{")) +
              boost::hana::unpack(std::make_index_sequence<cols>{},
                                  [&](auto... col) {
                                    return join(s_str(", "),
                                                fmt_v(val(row(), col()))...);
                                  }) +
              s(s_str("}"));
          return out;
        }(row)...);
      });
}
} // namespace detail
} // namespace eigen

template <typename Scalar, int rows, int cols>
struct FmtImpl<Eigen::Matrix<Scalar, rows, cols>> {
  PRINTF_DBG_HOST_DEVICE static auto
  fmt(const Eigen::Matrix<Scalar, rows, cols> &val) {
    return eigen::detail::eigen_fmt_impl<rows, cols>(val);
  }
};

template <typename Scalar, int rows, int cols>
struct FmtImpl<Eigen::Array<Scalar, rows, cols>> {
  PRINTF_DBG_HOST_DEVICE static auto
  fmt(const Eigen::Array<Scalar, rows, cols> &val) {
    return eigen::detail::eigen_fmt_impl<rows, cols>(val);
  }
};

template <typename Scalar, int dim, Eigen::TransformTraits type>
struct FmtImpl<Eigen::Transform<Scalar, dim, type>> {
  PRINTF_DBG_HOST_DEVICE static auto
  fmt(const Eigen::Transform<Scalar, dim, type> &val) {
    return fmt_v(val.matrix());
  }
};
} // namespace printf_dbg
