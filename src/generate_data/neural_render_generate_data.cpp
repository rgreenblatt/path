#include "generate_data/gen_data.h"

PYBIND11_MODULE(neural_render_generate_data, m) {
  m.def("gen_data", &generate_data::gen_data,
        "generate data in tensor form, output is scenes, coords, values ");
}
