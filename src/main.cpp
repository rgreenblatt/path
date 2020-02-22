#include "lib/cuda/utils.h"
#include "render/renderer.h"
#include "render/settings.h"
#include "scene/scenefile_compat/scenefile_loader.h"

#include <QImage>
#include <docopt.h>
#include <thrust/optional.h>

#include <string>
#include <iostream>

static const char USAGE[] =
    R"(Path

    Usage:
      path <scene_file> [-c | --cpu] [--width=<pixels>] [--height=<pixels>]
        [--samples=<count>] [--file=<file_name>]
      path (-h | --help)

    Options:
      -h --help           Show this screen.
      -c --cpu            Use cpu (default is to use gpu if found)
      --width=<pixels>    Width in pixels [default: 1024]
      --height=<pixels>   Height in pixels [default: 1024]
      --samples=<count>   Samples per pixel [default: 128]
      --file=<file_name>  File name [default: out.png]
)";

int main(int argc, char *argv[]) {
  const std::map<std::string, docopt::value> args =
      docopt::docopt(USAGE, {argv + 1, argv + argc});

  auto get_unpack_arg = [&](const std::string &s) {
    auto it = args.find(s);
    if (it == args.end()) {
      std::cerr << "internal command line parse error" << std::endl;
      abort();
    }

    return it->second;
  };

  bool using_cpu = get_unpack_arg("--cpu").asBool();
  const unsigned width = get_unpack_arg("--width").asLong();
  const unsigned height = get_unpack_arg("--height").asLong();
  const unsigned samples = get_unpack_arg("--samples").asLong();
  const std::string scene_file_name = get_unpack_arg("<scene_file>").asString();
  const std::string file_name = get_unpack_arg("--file").asString();

  if (!using_cpu) {
    int n_devices;

    CUDA_ERROR_CHK(cudaGetDeviceCount(&n_devices));
    for (int i = 0; i < n_devices; i++) {
      cudaDeviceProp prop;
      CUDA_ERROR_CHK(cudaGetDeviceProperties(&prop, i));
      std::cout << "found gpu: " << prop.name << std::endl;
    }

    if (n_devices == 0) {
      std::cout << "no gpu found, using cpu" << std::endl;
      using_cpu = true;
    }
  }

  ExecutionModel execution_model =
      using_cpu ? ExecutionModel::CPU : ExecutionModel::GPU;

  QImage image(width, height, QImage::Format_RGB32);

  scene::scenefile_compat::ScenefileLoader loader;

  auto scene = loader.load_scene(scene_file_name, float(width) / height);

  if (!scene.has_value()) {
    std::cerr << "failed to load scene" << std::endl;
    return 1;
  }

  render::Renderer renderer;

  Span<BGRA> pixels(reinterpret_cast<BGRA *>(image.bits()), width * height);

  render::Settings settings;

  settings.compile_time.light_sampler_type() =
      render::LightSamplerType::WeightedAABB;

  renderer.render(execution_model, pixels, *scene, samples, width, height,
                  settings, false);

  image.save(file_name.c_str());

  return 0;
}
