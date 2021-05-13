#pragma once

#include "lib/vector_type.h"

#include <tuple>

namespace generate_data {
std::tuple<VectorT<std::tuple<unsigned, unsigned>>,
           VectorT<std::tuple<float, float>>>
baryocentric_coords(unsigned width, unsigned height);
}
