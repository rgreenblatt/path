#include "generate_data/baryocentric_coords.h"

namespace generate_data {
std::tuple<VectorT<std::tuple<unsigned, unsigned>>,
           VectorT<std::tuple<float, float>>>
baryocentric_coords(unsigned width, unsigned height) {
  VectorT<std::tuple<unsigned, unsigned>> indexes;
  VectorT<std::tuple<float, float>> grid;

  for (unsigned y = 0; y < height; ++y) {
    for (unsigned x = 0; x < width; ++x) {
      float x_v = float(x + 1) / (width + 1);
      float y_v = float(y + 1) / (height + 1);

      if (x_v + y_v < 1.f) {
        indexes.push_back({x, y});
        grid.push_back({x_v, y_v});
      }
    }
  }

  return {indexes, grid};
}
} // namespace generate_data
