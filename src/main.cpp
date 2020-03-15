#include "lib/cuda/utils.h"
#include "render/renderer.h"
#include "render/settings.h"
#include "scene/scenefile_compat/scenefile_loader.h"

#include <QImage>
#include <docopt.h>
#include <magic_enum.hpp>
#include <thrust/optional.h>

#include <iostream>
#include <sstream>
#include <string>

static const char USAGE[] =
    R"(Path

    Usage:
      path <scene_file> [-g | --gpu] [--width=<pixels>] [--height=<pixels>]
        [--samples=<count>] [--file=<file_name>] [--uniform | --brdf]
        [--random-triangle | --no-light-sampling | --weighted-aabb]
        [--m-func-term-prob | --const-term-prob | --direct-only ]
      path (-h | --help)

    Options:
      -h --help             Show this screen.
      -g --gpu              Use gpu
      --width=<pixels>      Width in pixels [default: 1024]
      --height=<pixels>     Height in pixels [default: 1024]
      --samples=<count>     Samples per pixel [default: 128]
      --file=<file_name>    File name [default: out.png]

      --uniform             Uniform direction sampling
      --brdf                BRDF direction sampling (default)

      --random-triangle     Sample lights by randomly selecting an emissive
                            triangle weighting by the product of the surface
                            area and intensity of the triangle (default)
      --no-light-sampling   No direct lighting event splitting
      --weighted-aabb       Sample lights using light axis aligned bounding
                            boxs (incorrect (biased) and worse than random
                            triangle sampling)

      --m-func-term-prob    Term probability is a function of the multiplier
                            (default)
      --const-term-prob     Constant term probability
      --direct-only         Terminate after 1 iteration to just show
                            direct lighting (same as a constant
                            term probability with value 1)
)";

int main(int argc, char *argv[]) {
  const std::map<std::string, docopt::value> args =
      docopt::docopt(USAGE, {argv + 1, argv + argc});

  auto get_unpack_arg = [&](const std::string &s) {
    auto it = args.find(s);
    if (it == args.end()) {
      std::cerr << "internal command line parse error" << std::endl;
      std::cerr << s << std::endl;
      abort();
    }

    return it->second;
  };

  bool using_gpu = get_unpack_arg("--gpu").asBool();
  const unsigned width = get_unpack_arg("--width").asLong();
  const unsigned height = get_unpack_arg("--height").asLong();
  const unsigned samples = get_unpack_arg("--samples").asLong();
  const std::string scene_file_name = get_unpack_arg("<scene_file>").asString();
  const std::string file_name = get_unpack_arg("--file").asString();

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

  QImage image(width, height, QImage::Format_RGB32);

  scene::scenefile_compat::ScenefileLoader loader;

  auto scene = loader.load_scene(scene_file_name, float(width) / height);

  if (!scene.has_value()) {
    std::cerr << "failed to load scene" << std::endl;
    return 1;
  }

  render::Renderer renderer;

  Span<BGRA> pixels(reinterpret_cast<BGRA *>(image.bits()), width * height);

  using namespace magic_enum::ostream_operators;

  render::Settings settings;
  settings.compile_time.dir_sampler_type() =
      get_unpack_arg("--uniform").asBool() ? render::DirSamplerType::Uniform
                                           : render::DirSamplerType::BRDF;
  settings.compile_time.light_sampler_type() =
      get_unpack_arg("--weighted-aabb").asBool()
          ? render::LightSamplerType::WeightedAABB
          : (get_unpack_arg("--no-light-sampling").asBool()
                 ? render::LightSamplerType::NoLightSampling
                 : render::LightSamplerType::RandomTriangle);
  settings.compile_time.term_prob_type() =
      get_unpack_arg("--direct-only").asBool()
          ? render::TermProbType::DirectLightingOnly
          : (get_unpack_arg("--const-term-prob").asBool()
                 ? render::TermProbType::Constant
                 : render::TermProbType::MultiplierFunc);
  settings.compile_time.mesh_accel_type() = intersect::accel::AccelType::KDTree;
  settings.compile_time.triangle_accel_type() =
      intersect::accel::AccelType::KDTree;

  std::cout << "Direction sampling: "
            << settings.compile_time.dir_sampler_type() << std::endl;
  std::cout << "Light sampling: " << settings.compile_time.light_sampler_type()
            << std::endl;
  std::cout << "Term prob: " << settings.compile_time.term_prob_type()
            << std::endl;
  std::cout << "Rng: " << settings.compile_time.rng_type() << std::endl;

  std::ostringstream os;
  {
    cereal::YAMLOutputArchive archive(os);
    archive(CEREAL_NVP(settings));
  }
  std::cout << "output=\n" << os.str() << std::endl;

  renderer.render(execution_model, pixels, *scene, samples, width, height,
                  settings, false);

  image.save(file_name.c_str());

  return 0;
}
