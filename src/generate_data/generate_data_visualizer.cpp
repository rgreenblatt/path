#include "execution_model/execution_model.h"
#include "generate_data/amend_config.h"
#include "generate_data/baryocentric_coords.h"
#include "generate_data/baryocentric_to_ray.h"
#include "generate_data/generate_scene.h"
#include "generate_data/generate_scene_triangles.h"
#include "generate_data/normalize_scene_triangles.h"
#include "integrate/sample_triangle.h"
#include "intersect/triangle.h"
#include "lib/assert.h"
#include "lib/cuda/utils.h"
#include "render/config_io.h"
#include "render/renderer.h"
#include "rng/uniform/uniform.h"
#include "scene/camera.h"

#include <Eigen/Dense>
#include <QImage>
#include <boost/lexical_cast.hpp>
#include <docopt.h>

#include <iostream>
#include <random>
#include <string>

#include "dbg.h"

// In retrospect, I don't really like docopt...
constexpr char USAGE[] =
    R"(Path

    Usage:
      generate_data_visualizer [--seed=<seed>] [--config=<file_name>]
        [-g | --gpu] [--print-config]
      generate_data_visualizer (-h | --help)

    Options:
      -h --help                  Show this screen.
      --seed=<seed>              Random seed [default: 0]
      --config=<file_name>       Config file name. If no file is specified,
                                 default settings will be used.
      -g --gpu                   Use gpu
      --print-config             Print config
)";

int main(int argc, char *argv[]) {
  using namespace generate_data;

  const std::map<std::string, docopt::value> args =
      docopt::docopt(USAGE, {argv + 1, argv + argc});

  auto get_unpack_arg = [&](const std::string &s) {
    auto it = args.find(s);
    if (it == args.end()) {
      std::cerr << "internal command line parse error" << std::endl;
      std::cerr << s << std::endl;
      unreachable();
    }

    return it->second;
  };

  bool using_gpu = get_unpack_arg("--gpu").asBool();
  const std::string output_file_name = "out.png";
  const std::string baryocentric_output_file_name = "baryo.png";
  const bool print_config = get_unpack_arg("--print-config").asBool();
  const unsigned seed = get_unpack_arg("--seed").asLong();

  if (using_gpu) {
    int n_devices;

    CUDA_ERROR_CHK(cudaGetDeviceCount(&n_devices));
    for (int i = 0; i < n_devices; i++) {
      cudaDeviceProp prop;
      CUDA_ERROR_CHK(cudaGetDeviceProperties(&prop, i));
      std::cout << "found gpu: " << prop.name << std::endl;
    }

    if (n_devices == 0) {
      std::cout << "no gpu found, using cpu" << std::endl;
      using_gpu = false;
    }
  }

  ExecutionModel execution_model =
      using_gpu ? ExecutionModel::GPU : ExecutionModel::CPU;

  UniformState rng{seed};

  // auto tris = generate_scene_triangles(rng);
  auto tris = normalize_scene_triangles(generate_scene_triangles(rng));
  auto scene = generate_scene(tris);

  auto dir_towards = -tris.triangle_onto.template cast<float>().normal();

  Eigen::Vector3f onto_centroid =
      tris.triangle_onto.centroid().template cast<float>();

  auto film_to_world = scene::get_camera_transform(
      dir_towards, UnitVector::new_normalize({0.f, 1.f, 0.f}),
      onto_centroid - 8 * (*dir_towards), 45.f, 1.f);

  render::Renderer renderer;

  unsigned width = 256;
  unsigned height = width;
  unsigned num_samples = 1024;

  QImage image(width, height, QImage::Format_RGB32);
  Span<BGRA32> pixels(reinterpret_cast<BGRA32 *>(image.bits()), width * height);

  render::Settings settings;
  auto config_file_name = get_unpack_arg("--config");
  if (config_file_name) {
    settings = render::load_config(config_file_name.asString());
  }
  amend_config(settings);
  if (print_config) {
    render::print_config(settings);
  }

  renderer.render(execution_model,
                  {tag_v<render::SampleSpecType::SquareImage>,
                   {
                       .x_dim = width,
                       .y_dim = height,
                       .film_to_world = film_to_world,
                   }},
                  {tag_v<render::OutputType::BGRA>, pixels}, scene, num_samples,
                  settings, true);

  image.save(output_file_name.c_str());

  unsigned baryocentric_width = 128;
  unsigned baryocentric_height = baryocentric_width;
  unsigned baryocentric_num_samples = 4096;

  auto [baryocentric_indexes, baryocentric_grid_values] =
      baryocentric_coords(baryocentric_width, baryocentric_height);

  VectorT<intersect::Ray> baryocentric_grid(baryocentric_grid_values.size());

  for (unsigned i = 0; i < baryocentric_grid_values.size(); ++i) {
    auto [x_v, y_v] = baryocentric_grid_values[i];
    baryocentric_grid[i] = baryocentric_to_ray(
        x_v, y_v, tris.triangle_onto.template cast<float>(), dir_towards);
  }

  std::vector<BGRA32> baryocentric_pixels(baryocentric_grid.size(),
                                          BGRA32::Zero());

  renderer.render(
      execution_model,
      {tag_v<render::SampleSpecType::InitialRays>, baryocentric_grid},
      {tag_v<render::OutputType::BGRA>, baryocentric_pixels}, scene,
      baryocentric_num_samples, settings, true);

  QImage baryocentric_image(baryocentric_width, baryocentric_height,
                            QImage::Format_RGB32);
  SpanSized<BGRA32> baryocentric_image_pixels(
      reinterpret_cast<BGRA32 *>(baryocentric_image.bits()),
      baryocentric_width * baryocentric_height);
  std::fill(baryocentric_image_pixels.begin(), baryocentric_image_pixels.end(),
            BGRA32::Zero());
  for (unsigned i = 0; i < baryocentric_grid.size(); ++i) {
    auto [x, y] = baryocentric_indexes[i];
    baryocentric_image_pixels[x + y * baryocentric_width] =
        baryocentric_pixels[i];
  }
  baryocentric_image.save(baryocentric_output_file_name.c_str());

  return 0;
}
