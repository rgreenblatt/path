#pragma once

#include "execution_model/execution_model_vector_type.h"
#include "integrate/ray_info.h"
#include "integrate/rendering_equation_state.h"
#include "rng/rng.h"

namespace render {
namespace detail {
template <ExecutionModel exec, unsigned max_num_light_samples, rng::RngRef R>
struct IntegrateImageBulkState {
  using State = integrate::RenderingEquationState<max_num_light_samples>;
  ExecVector<exec, std::optional<State>> op_state;
  ExecVector<exec, integrate::FRayInfo> initial_sample_info;
  ExecVector<exec, intersect::Ray> rays;
  ExecVector<exec, std::optional<intersect::Ray>> op_rays;
  ExecVector<exec, ArrayVec<intersect::Ray, State::max_num_samples>>
      sample_rays;
  ExecVector<exec, typename R::State> rng_state;
  // TODO: add things as needed for bulk approaches
};
} // namespace detail
} // namespace render
