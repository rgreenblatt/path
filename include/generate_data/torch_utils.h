#pragma once

#include <ATen/Tensor.h>
#include <ATen/TensorOptions.h>

namespace generate_data {
using TorchIdxT = int64_t;

template <typename T> inline auto get_tensor_type() {
  return at::TensorOptions(caffe2::TypeMeta::Make<T>());
}

const static auto long_tensor_type = get_tensor_type<TorchIdxT>();
const static auto float_tensor_type = get_tensor_type<float>();
} // namespace generate_data
