#pragma once

#include "execution_model/execution_model_vector_type.h"
#include "integrate/rendering_equation_state.h"
#include "rng/rng.h"

namespace render {
namespace detail {
template <ExecutionModel exec, unsigned max_num_light_samples, rng::RngRef R>
struct IntegrateImageBulkState {
  using State = integrate::RenderingEquationState<max_num_light_samples>;
  ExecVector<exec, State> state;
  ExecVector<exec, Optional<State>> op_state;
  ExecVector<exec, typename R::State> rng_state;
};
} // namespace detail
} // namespace render
