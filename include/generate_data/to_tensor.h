#pragma once

#include "generate_data/torch_utils.h"

#include <ATen/ATen.h>
#include <boost/multi_array.hpp>

namespace generate_data {
template <typename T, size_t n>
at::Tensor to_tensor(boost::multi_array<T, n> &arr) {
  std::array<TorchIdxT, n> dims;
  std::copy(arr.shape(), arr.shape() + n, dims.begin());

  return at::from_blob(arr.data(), dims, get_tensor_type<T>()).clone();
}
} // namespace generate_data
