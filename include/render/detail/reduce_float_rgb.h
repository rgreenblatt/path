#pragma once

#include "execution_model/execution_model.h"
#include "execution_model/execution_model_vector_type.h"
#include "lib/bgra_32.h"
#include "lib/float_rgb.h"
#include "lib/span.h"

namespace render {
namespace detail {
// TODO: division/settings?
template <ExecutionModel exec> struct ReduceFloatRGB {
  static ExecVector<exec, FloatRGB> *
  run(bool output_as_bgra_32, unsigned reduction_factor, unsigned samples_per,
      ExecVector<exec, FloatRGB> *float_rgb_in,
      ExecVector<exec, FloatRGB> *float_rgb_out, Span<BGRA32> bgras);
};
} // namespace detail
} // namespace render
