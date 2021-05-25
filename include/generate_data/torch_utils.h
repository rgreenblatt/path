#pragma once

#include <ATen/Tensor.h>
#include <ATen/TensorOptions.h>

namespace generate_data {
using TorchIdxT = int64_t;
const static auto long_tensor_type =
    at::TensorOptions(caffe2::TypeMeta::Make<TorchIdxT>());
} // namespace generate_data
