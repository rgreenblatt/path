#include "generate_data/full_scene/scene_generator.h"
#include "lib/assert.h"
#include "lib/span.h"
#include "render/config_io.h"
#include "render/renderer.h"
#include "scene/camera.h"

#include <QImage>
#include <docopt.h>

#include <map>
#include <random>
#include <string>

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
  using namespace generate_data::full_scene;

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

  SceneGenerator generator;
  std::mt19937 rng(seed);
  const auto &scene = generator.generate(rng);

  auto film_to_world = scene::get_camera_transform(
      UnitVector::new_normalize({0.f, 0.f, -1.f}),
      UnitVector::new_normalize({0.f, 1.f, 0.f}), {0.f, 0.f, 10.f}, 45.f, 1.f);

  render::Renderer renderer;

  unsigned width = 256;
  unsigned height = width;
  unsigned num_samples = 8192;

  QImage image(width, height, QImage::Format_RGB32);
  Span<BGRA32> pixels(reinterpret_cast<BGRA32 *>(image.bits()), width * height);

  render::Settings settings;
  auto config_file_name = get_unpack_arg("--config");
  if (config_file_name) {
    settings = render::load_config(config_file_name.asString());
  }
  settings.rendering_equation_settings.back_cull_emission = false;
  // amend_config(settings);
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
}
