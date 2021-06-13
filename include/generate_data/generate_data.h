#pragma once

#include <ATen/Tensor.h>

namespace generate_data {
template <typename Inputs> struct StandardData {
  Inputs inputs;
  at::Tensor values;

  inline StandardData to(const at::Tensor &example_tensor) const {
    return {
        .inputs = inputs.to(example_tensor),
        .values = values.to(example_tensor.device(), example_tensor.dtype()),
    };
  }
};

template <typename Inputs> struct ImageData {
  StandardData<Inputs> standard;
  at::Tensor image_indexes;

  inline ImageData to(const at::Tensor &example_tensor) const {
    return {
        .standard = standard.to(example_tensor),
        .image_indexes = image_indexes.to(example_tensor.device()),
    };
  }
};
} // namespace generate_data
