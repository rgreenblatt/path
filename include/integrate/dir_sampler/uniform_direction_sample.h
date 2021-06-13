#pragma once

#include "lib/cuda/utils.h"
#include "lib/projection.h"
#include "lib/unit_vector.h"
#include "rng/rng.h"

namespace integrate {
namespace dir_sampler {
template <rng::RngState R>
HOST_DEVICE UnitVector uniform_direction_sample(R &rng,
                                                const UnitVector &relative_to,
                                                bool need_whole_sphere) {
  float v0 = rng.next();
  float v1 = rng.next();

  float phi = 2 * M_PI * v0;
  float theta = std::acos(need_whole_sphere ? 2 * v1 - 1 : v1);

  auto direction = find_relative_vec(relative_to, phi, theta);

  debug_assert(need_whole_sphere || direction->dot(*relative_to) >= 0);

  return direction;
}
} // namespace dir_sampler
} // namespace integrate
