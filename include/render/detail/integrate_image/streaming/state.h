#pragma once

#include "execution_model/execution_model_single_item.h"
#include "execution_model/execution_model_vector_type.h"
#include "integrate/ray_info.h"
#include "integrate/rendering_equation_state.h"
#include "kernel/atomic.h"
#include "rng/rng.h"

namespace render {
namespace detail {
namespace integrate_image {
namespace streaming {
template <unsigned max_num_light_samples> struct SampleState {
  integrate::RenderingEquationState<max_num_light_samples> render_state;
  unsigned ray_idx;
  unsigned sample_idx;
  unsigned x;
  unsigned y;
};

struct InitialSampleInfo {
  integrate::FRayInfo ray_info;
  unsigned sample_idx;
  unsigned x;
  unsigned y;
};

template <ExecutionModel exec, unsigned max_num_light_samples, rng::RngRef R>
struct State {
  std::array<ExecVector<exec, SampleState<max_num_light_samples>>, 2>
      states_in_and_out;
  std::array<ExecVector<exec, typename R::SavedState>, 2> rng_states_in_and_out;
  ExecVector<exec, InitialSampleInfo> initial_sample_info;
  std::array<ExecVector<exec, intersect::Ray>, 2> rays_in_and_out;
  ExecVector<exec, std::array<kernel::Atomic<exec, float>, 3>> float_rgb;

  ExecSingleItem<exec, kernel::Atomic<exec, unsigned>> state_idx = {0};
  ExecSingleItem<exec, kernel::Atomic<exec, unsigned>> ray_idx = {0};

  // TODO: add things as needed for bulk approaches
  //
};
} // namespace streaming
} // namespace integrate_image
} // namespace detail
} // namespace render
