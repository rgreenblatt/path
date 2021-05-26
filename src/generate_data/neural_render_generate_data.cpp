#include "generate_data/constants.h"
#include "generate_data/gen_data.h"

#include <torch/extension.h>

PYBIND11_MODULE(neural_render_generate_data, m) {
  m.def("gen_data", &generate_data::gen_data,
        "generate data in tensor form, output is scenes, coords, values");
  m.def("gen_data_for_image", &generate_data::gen_data_for_image,
        "generate data in tensor form, output is scenes, coords, values, "
        "indexes");
  m.def("deinit_renderers", &generate_data::deinit_renderers,
        "destroy renderers");
  py::class_<generate_data::PolygonInput>(m, "PolygonInput")
      .def_readwrite("point_values", &generate_data::PolygonInput::point_values)
      .def_readwrite("overall_features",
                     &generate_data::PolygonInput::overall_features)
      .def_readwrite("counts", &generate_data::PolygonInput::counts)
      .def_readwrite("prefix_sum_counts",
                     &generate_data::PolygonInput::prefix_sum_counts)
      .def_readwrite("item_to_left_idxs",
                     &generate_data::PolygonInput::item_to_left_idxs)
      .def_readwrite("item_to_right_idxs",
                     &generate_data::PolygonInput::item_to_right_idxs)
      .def("to", &generate_data::PolygonInput::to);
  py::class_<generate_data::PolygonInputForTri>(m, "PolygonInputForTri")
      .def_readwrite("polygon_feature",
                     &generate_data::PolygonInputForTri::polygon_feature)
      .def_readwrite("tri_idx", &generate_data::PolygonInputForTri::tri_idx)
      .def("to", &generate_data::PolygonInputForTri::to);
  py::class_<generate_data::RayInput>(m, "RayInput")
      .def_readwrite("values", &generate_data::RayInput::values)
      .def_readwrite("counts", &generate_data::RayInput::counts)
      .def_readwrite("prefix_sum_counts",
                     &generate_data::RayInput::prefix_sum_counts)
      .def_readwrite("is_ray", &generate_data::RayInput::is_ray)
      .def("to", &generate_data::RayInput::to);
  py::class_<generate_data::NetworkInputs>(m, "NetworkInputs")
      .def_readwrite("overall_scene_features",
                     &generate_data::NetworkInputs::overall_scene_features)
      .def_readwrite("triangle_features",
                     &generate_data::NetworkInputs::triangle_features)
      .def_readwrite("polygon_inputs",
                     &generate_data::NetworkInputs::polygon_inputs)
      .def_readwrite("ray_inputs", &generate_data::NetworkInputs::ray_inputs)
      .def_readwrite("baryocentric_coords",
                     &generate_data::NetworkInputs::baryocentric_coords)
      .def("to", &generate_data::NetworkInputs::to);
  py::class_<generate_data::StandardData>(m, "StandardData")
      .def_readwrite("inputs", &generate_data::StandardData::inputs)
      .def_readwrite("values", &generate_data::StandardData::values)
      .def("to", &generate_data::StandardData::to);
  py::class_<generate_data::ImageData>(m, "ImageData")
      .def_readwrite("standard", &generate_data::ImageData::standard)
      .def_readwrite("image_indexes", &generate_data::ImageData::image_indexes)
      .def("to", &generate_data::ImageData::to);
  py::class_<generate_data::Constants>(m, "Constants")
      .def(py::init<>())
      .def_readwrite("n_tris", &generate_data::Constants::n_tris)
      .def_readwrite("n_scene_values",
                     &generate_data::Constants::n_scene_values)
      .def_readwrite("n_dims", &generate_data::Constants::n_dims)
      .def_readwrite("n_tri_values", &generate_data::Constants::n_tri_values)
      .def_readwrite("n_baryo_dims", &generate_data::Constants::n_baryo_dims)
      .def_readwrite("n_coords_feature_values",
                     &generate_data::Constants::n_coords_feature_values)
      .def_readwrite("n_poly_point_values",
                     &generate_data::Constants::n_poly_point_values)
      .def_readwrite("n_rgb_dims", &generate_data::Constants::n_rgb_dims)
      .def_readwrite("n_shadowable_tris",
                     &generate_data::Constants::n_shadowable_tris)
      .def_readwrite("n_poly_feature_values",
                     &generate_data::Constants::n_poly_feature_values)
      .def_readwrite("n_polys", &generate_data::Constants::n_polys)
      .def_readwrite("n_ray_item_values",
                     &generate_data::Constants::n_ray_item_values)
      .def_readwrite("n_ray_items", &generate_data::Constants::n_ray_items);
}
