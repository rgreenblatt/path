#pragma once

#include "integrate/ray_info.h"
#include "lib/array_vec.h"
#include "lib/float_rgb.h"
#include "lib/span.h"
#include "lib/tagged_union.h"
#include "meta/all_values/impl/enum.h"

#include <concepts>

namespace integrate {
// this should probably be a class with a friend struct...
template <unsigned max_num_light_samples> struct RenderingEquationState {
  HOST_DEVICE static RenderingEquationState
  initial_state(const FRayInfo &ray_info) {
    return RenderingEquationState{
        .iters = 0,
        .count_emission = true,
        .has_next_sample = true,
        .ray_info = {FloatRGB::Constant(ray_info.multiplier),
                     ray_info.target_distance},
        .light_samples = {},
        .float_rgb_total = FloatRGB::Zero(),
    };
  }

  static constexpr unsigned max_num_samples = max_num_light_samples + 1;

  unsigned iters;
  bool count_emission;
  bool has_next_sample;
  ArrRayInfo ray_info;
  ArrayVec<ArrRayInfo, max_num_light_samples> light_samples;
  FloatRGB float_rgb_total;

  HOST_DEVICE unsigned num_samples() const {
    return light_samples.size() + has_next_sample;
  }

  // handle things this way so the caller is flexible in how they use the
  // iteration and what state they maintain
  HOST_DEVICE std::optional<intersect::Ray>
  get_last_ray(Span<const intersect::Ray> rays) const {
    if (has_next_sample) {
      rays.debug_assert_size_is(num_samples());
      return rays[num_samples() - 1];
    } else {
      return std::nullopt;
    }
  }
};

static_assert(std::semiregular<RenderingEquationState<0>>);
static_assert(std::semiregular<RenderingEquationState<3>>);

template <unsigned max_num_light_samples>
struct RenderingEquationNextIteration {
  RenderingEquationState<max_num_light_samples> state;
  ArrayVec<intersect::Ray, max_num_light_samples + 1> rays;
};

enum class IterationOutputType {
  NextIteration,
  Finished,
};

template <unsigned max_num_light_samples>
using IterationOutput =
    TaggedUnion<IterationOutputType,
                RenderingEquationNextIteration<max_num_light_samples>,
                FloatRGB>;
} // namespace integrate
