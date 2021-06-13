#include "generate_data/single_triangle/constants.h"
#include "generate_data/single_triangle/generate_data.h"

#include <torch/extension.h>

using namespace generate_data;
using namespace generate_data::single_triangle;

PYBIND11_MODULE(neural_render_generate_data_single_triangle, m) {
  m.def("generate_data", &generate_data::single_triangle::generate_data,
        "generate data in tensor form, output is scenes, coords, values");
  m.def("generate_data_for_image", &generate_data_for_image,
        "generate data in tensor form, output is scenes, coords, values, "
        "indexes");
  m.def("deinit_renderers", &deinit_renderers, "destroy renderers");
  py::class_<PolygonInput>(m, "PolygonInput")
      .def_readwrite("point_values", &PolygonInput::point_values)
      .def_readwrite("overall_features", &PolygonInput::overall_features)
      .def_readwrite("counts", &PolygonInput::counts)
      .def_readwrite("prefix_sum_counts", &PolygonInput::prefix_sum_counts)
      .def_readwrite("item_to_left_idxs", &PolygonInput::item_to_left_idxs)
      .def_readwrite("item_to_right_idxs", &PolygonInput::item_to_right_idxs)
      .def("to", &PolygonInput::to);
  py::class_<PolygonInputForTri>(m, "PolygonInputForTri")
      .def_readwrite("polygon_feature", &PolygonInputForTri::polygon_feature)
      .def_readwrite("tri_idx", &PolygonInputForTri::tri_idx)
      .def("to", &PolygonInputForTri::to);
  py::class_<RayInput>(m, "RayInput")
      .def_readwrite("values", &RayInput::values)
      .def_readwrite("counts", &RayInput::counts)
      .def_readwrite("prefix_sum_counts", &RayInput::prefix_sum_counts)
      .def_readwrite("is_ray", &RayInput::is_ray)
      .def("to", &RayInput::to);
  py::class_<NetworkInputs>(m, "NetworkInputs")
      .def_readwrite("overall_scene_features",
                     &NetworkInputs::overall_scene_features)
      .def_readwrite("triangle_features", &NetworkInputs::triangle_features)
      .def_readwrite("polygon_inputs", &NetworkInputs::polygon_inputs)
      .def_readwrite("ray_inputs", &NetworkInputs::ray_inputs)
      .def_readwrite("baryocentric_coords", &NetworkInputs::baryocentric_coords)
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
  using Const = generate_data::single_triangle::Constants;
  py::class_<Const>(m, "Constants")
      .def(py::init<>())
      .def_readwrite("n_tris", &Const::n_tris)
      .def_readwrite("n_scene_values", &Const::n_scene_values)
      .def_readwrite("n_dims", &Const::n_dims)
      .def_readwrite("n_tri_values", &Const::n_tri_values)
      .def_readwrite("n_baryo_dims", &Const::n_baryo_dims)
      .def_readwrite("n_coords_feature_values", &Const::n_coords_feature_values)
      .def_readwrite("n_poly_point_values", &Const::n_poly_point_values)
      .def_readwrite("n_rgb_dims", &Const::n_rgb_dims)
      .def_readwrite("n_shadowable_tris", &Const::n_shadowable_tris)
      .def_readwrite("n_poly_feature_values", &Const::n_poly_feature_values)
      .def_readwrite("n_polys", &Const::n_polys)
      .def_readwrite("n_ray_item_values", &Const::n_ray_item_values)
      .def_readwrite("n_ray_items", &Const::n_ray_items);
}
