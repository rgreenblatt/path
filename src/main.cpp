#include "lib/info/timer.h"
#include "render/renderer.h"
#include "render/settings.h"
#include "scene/scenefile_compat/scenefile_loader.h"

#include <QImage>
#include <boost/lexical_cast.hpp>
#include <docopt.h>
#include <magic_enum.hpp>
#include <thrust/optional.h>

#include <fstream>
#include <iostream>
#include <string>

#include "lib/info/debug_print.h"

// In retrospect, I don't really like docopt...
static const char USAGE[] =
    R"(Path

    Usage:
      path <scene_file> [--width=<pixels>] [--height=<pixels>]
        [--samples=<count>] [--output=<file_name>] [--config-file=<file_name>]
        [-g | --gpu] [--profile-samples] [--samples-min=<min>]
        [--samples-max=<max>] [--bench] [--bench-budget=<time>]
        [--disable-progress]
      path (-h | --help)

    Options:
      -h --help             Show this screen.
      --width=<pixels>      Width in pixels [default: 1024]
      --height=<pixels>     Height in pixels [default: 1024]
      --samples=<count>     Samples per pixel [default: 128]
      --output=<file_name>  File name [default: out.png]
      --config=<file_name>  Config file name. If no file is specified, default
                            settings will be used.
      -g --gpu              Use gpu

      --profile-samples     Run sample profiling (compute mean absolute pixel
                            error and time)
      --samples-min=<min>   Min samples for profiling [default: 1]
      --samples-max=<max>   Max samples for profiling [default: 4096]

      --bench               Warm up and then run multiple times and report 
                            statistics
      --bench-budget=<time> Approximate time in seconds for bench [default: 5.0]
      --disable-progress         Disable progress bar
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
  const auto scene_file_name = get_unpack_arg("<scene_file>").asString();
  const auto output_file_name = get_unpack_arg("--output").asString();
  const bool disable_progress = get_unpack_arg("--disable-progress").asBool();

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

  render::Settings settings;

  auto config_file_name = get_unpack_arg("--config-file");
  if (config_file_name) {
    std::ifstream i(config_file_name.asString(), std::ifstream::binary);
    cereal::YAMLInputArchive archive(i);
    archive(settings);
  }

  std::ostringstream os;
  {
    cereal::YAMLOutputArchive archive(os);
    archive(CEREAL_NVP(settings));
  }
  std::cout << os.str() << std::endl;

  unsigned updated_samples = samples;

  auto render = [&](bool show_progress, bool show_times) {
    renderer.render(execution_model, pixels, *scene, updated_samples, width,
                    height, settings, show_progress && !disable_progress,
                    show_times);
  };

  if (get_unpack_arg("--bench").asBool()) {
    const unsigned warmup_iters = 2;

    Timer total_warm_up;

    auto b_render = [&] { render(false, false); };

    b_render();

    Timer warmup_time;

    for (unsigned i = 0; i < warmup_iters; i++) {
      b_render();
    }

    float time_per = warmup_time.elapsed() / warmup_iters;

    unsigned iters = unsigned(std::ceil(
        std::max(0.f, boost::lexical_cast<float>(
                          get_unpack_arg("--bench-budget").asString()) /
                          time_per)));

    if (iters < 3) {
      std::cerr << "Note that only " << iters
                << " iter(s) can be run with the current budget" << std::endl;
    }

    float mean_time = 0.f;

    for (unsigned i = 0; i < iters; ++i) {
      Timer render_timer;
      b_render();
      mean_time += render_timer.elapsed();
    }

    mean_time /= iters;

    std::cout << "mean: " << mean_time << std::endl;

  } else {
    render(true, false);
  }

  if (updated_samples != samples) {
    std::cout << "samples changed from " << samples << " to " << updated_samples
              << std::endl;
  }

  image.save(output_file_name.c_str());

  return 0;
}
