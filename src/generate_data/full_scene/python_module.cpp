#include "generate_data/full_scene/constants.h"
#include "generate_data/full_scene/generate_data.h"

#include <torch/extension.h>

using namespace generate_data;
using namespace generate_data::full_scene;

PYBIND11_MODULE(neural_render_generate_data_full_scene, m) {
  m.def("generate_data", &generate_data::full_scene::generate_data,
        "generate data in tensor form, output is scenes, coords, values");
  m.def("generate_data_for_image", &generate_data_for_image,
        "generate data in tensor form, output is scenes, coords, values, "
        "indexes");
  m.def("deinit_renderers", &deinit_renderers, "destroy renderers");
  using Const = generate_data::full_scene::Constants;
  py::class_<Const>(m, "Constants")
      .def(py::init<>())
      .def_readwrite("n_dims", &Const::n_dims)
      .def_readwrite("n_tri_values", &Const::n_tri_values)
      .def_readwrite("n_baryo_dims", &Const::n_baryo_dims)
      .def_readwrite("n_coords_feature_values", &Const::n_coords_feature_values)
      .def_readwrite("n_rgb_dims", &Const::n_rgb_dims)
      .def_readwrite("n_bsdf_values", &Const::n_bsdf_values);
  py::class_<NetworkInputs>(m, "NetworkInputs")
      .def_readwrite("triangle_features", &NetworkInputs::triangle_features)
      .def_readwrite("mask", &NetworkInputs::mask)
      .def_readwrite("bsdf_features", &NetworkInputs::bsdf_features)
      .def_readwrite("emissive_values", &NetworkInputs::emissive_values)
      .def_readwrite("baryocentric_coords", &NetworkInputs::baryocentric_coords)
      .def_readwrite("triangle_idxs_for_coords",
                     &NetworkInputs::triangle_idxs_for_coords)
      .def_readwrite("total_tri_count", &NetworkInputs::total_tri_count)
      .def_readwrite("n_samples_per", &NetworkInputs::n_samples_per)
      .def("to", &NetworkInputs::to);
  using Stand = StandardData<NetworkInputs>;
  py::class_<Stand>(m, "StandardData")
      .def_readwrite("inputs", &Stand::inputs)
      .def_readwrite("values", &Stand::values)
      .def("to", &Stand::to);
  using Image = ImageData<NetworkInputs>;
  py::class_<Image>(m, "ImageData")
      .def_readwrite("standard", &Image::standard)
      .def_readwrite("image_indexes", &Image::image_indexes)
      .def("to", &Image::to);
}
