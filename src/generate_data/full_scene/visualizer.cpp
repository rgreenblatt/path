#include "generate_data/full_scene/amend_config.h"
#include "generate_data/full_scene/default_film_to_world.h"
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
      generate_data_visualizer [--seed=<seed>] [--n-output-steps=<steps>]
        [--config=<file_name>] [-g | --gpu] [--print-config]
      generate_data_visualizer (-h | --help)

    Options:
      -h --help                 Show this screen.
      --seed=<seed>             Random seed [default: 0]
      --n-output-steps=<steps>  Number of steps for output images [default: 8]
      --config=<file_name>      Config file name. If no file is specified,
                                default settings will be used.
      -g --gpu                  Use gpu
      --print-config            Print config
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
  const unsigned n_output_steps = get_unpack_arg("--n-output-steps").asLong();
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
  const auto &scene = std::get<0>(generator.generate(rng));

  auto film_to_world = default_film_to_world();

  render::Renderer renderer;

  unsigned width = 256;
  unsigned height = width;
  unsigned num_samples = 8192;

  render::Settings settings;
  auto config_file_name = get_unpack_arg("--config");
  if (config_file_name) {
    settings = render::load_config(config_file_name.asString());
  }
  amend_config(settings);
  if (print_config) {
    render::print_config(settings);
  }

  VectorT<VectorT<FloatRGB>> step_outputs(n_output_steps,
                                          VectorT<FloatRGB>{width * height});
  VectorT<Span<FloatRGB>> outputs(step_outputs.begin(), step_outputs.end());

  renderer.render(execution_model,
                  {tag_v<render::SampleSpecType::SquareImage>,
                   {
                       .x_dim = width,
                       .y_dim = height,
                       .film_to_world = film_to_world,
                   }},
                  {tag_v<render::OutputType::OutputPerStep>, outputs}, scene,
                  num_samples, settings, true);
  for (unsigned i = 0; i < step_outputs.size(); ++i) {
    QImage image(width, height, QImage::Format_RGB32);
    std::transform(step_outputs[i].begin(), step_outputs[i].end(),
                   reinterpret_cast<BGRA32 *>(image.bits()),
                   [&](const FloatRGB &v) { return float_rgb_to_bgra_32(v); });
    std::stringstream name_s;
    name_s << "out_" << i << ".png";
    image.save(name_s.str().c_str());
  }

  QImage summed_image(width, height, QImage::Format_RGB32);
  SpanSized<BGRA32> vals{reinterpret_cast<BGRA32 *>(summed_image.bits()),
                         width * height};
  for (unsigned i = 0; i < vals.size(); ++i) {
    FloatRGB total = FloatRGB::Zero();

    for (unsigned j = 0; j < step_outputs.size(); ++j) {
      total += step_outputs[j][i];
    }

    vals[i] = float_rgb_to_bgra_32(total);
  }
  summed_image.save(output_file_name.c_str());
}
