#pragma once

#include <Eigen/Core>

namespace cereal {
template <typename Archive> void serialize(Archive &ar, Eigen::Array3f &arr) {
  ar(arr[0], arr[1], arr[2]);
}
} // namespace cereal
