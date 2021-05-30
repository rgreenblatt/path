#include "lib/assert.h"
#include "lib/cuda/utils.h"
#include "lib/info/timer.h"
#include "render/renderer_from_files.h"

#include <QImage>
#include <boost/lexical_cast.hpp>
#include <docopt.h>

#include <iostream>
#include <string>

#include "dbg.h"

// In retrospect, I don't really like docopt...
constexpr char USAGE[] =
    R"(Path

    Usage:
      path <scene_file> [--width=<pixels>] [--height=<pixels>]
        [--samples=<count>] [--output=<file_name>] [--n-output-steps=<steps>]
        [--output-per-step] [--bench-budget=<time>] [--config=<file_name>]
        [-g | --gpu] [--bench] [--bench-ignore-accel] [--disable-progress]
        [--show-times] [--no-print-config]
      path (-h | --help)

    Options:
      -h --help                 Show this screen.
      --width=<pixels>          Width in pixels [default: 1024]
      --height=<pixels>         Height in pixels [default: 1024]
      --samples=<count>         Samples per pixel [default: 128]
      --output=<file_name>      File name [default: out.png]
      --output-per-step         Create output image per each step
      --n-output-steps=<steps>  Number of steps for output images [default: 8]
      --bench-budget=<time>     Approx time in seconds for bench [default: 5.0]
      --config=<file_name>      Config file name. If no file is specified,
                                default settings will be used.
      -g --gpu                  Use gpu

      --bench                   Warm up and then run multiple times and report
                                statistics
      --bench-ignore-accel      Ignore the computation used to build the accel
      --disable-progress        Disable progress bar
      --show-times              Show timings
      --no-print-config         Don't print config
)";

int main(int argc, char *argv[]) {
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
  const unsigned width = get_unpack_arg("--width").asLong();
  const unsigned height = get_unpack_arg("--height").asLong();
  const unsigned samples = get_unpack_arg("--samples").asLong();
  const auto scene_file_name = get_unpack_arg("<scene_file>").asString();
  const auto output_file_name = get_unpack_arg("--output").asString();
  const bool disable_progress = get_unpack_arg("--disable-progress").asBool();
  const bool show_times = get_unpack_arg("--show-times").asBool();
  const bool print_config = !get_unpack_arg("--no-print-config").asBool();

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

  render::RendererFromFiles renderer;

  renderer.load_scene(scene_file_name, float(width) / height);
  auto config_file_name = get_unpack_arg("--config");
  if (config_file_name) {
    renderer.load_config(config_file_name.asString());
  }
  if (print_config) {
    renderer.print_config();
  }

  Span<BGRA32> pixels(reinterpret_cast<BGRA32 *>(image.bits()), width * height);

  auto render_gen = [&](bool show_progress, const render::Output &output) {
    return renderer.render(execution_model, width, height, output, samples,
                           show_progress && !disable_progress, show_times);
  };
  auto render = [&](bool show_progress) {
    return render_gen(show_progress,
                      render::Output{tag_v<render::OutputType::BGRA>, pixels});
  };

  bool is_output_per_step = get_unpack_arg("--output-per-step").asBool();
  const unsigned n_output_steps = get_unpack_arg("--n-output-steps").asLong();

  if (get_unpack_arg("--bench").asBool()) {
    const unsigned warmup_iters = 2;

    Timer total_warm_up;

    auto b_render = [&] { return render(false); };

    b_render();

    Timer warmup_time;
    double total_warm_up_time_ignore_accel = 0.f;

    for (unsigned i = 0; i < warmup_iters; i++) {
      total_warm_up_time_ignore_accel += b_render();
    }

    double time_per = warmup_time.elapsed() / warmup_iters;

    unsigned iters = unsigned(std::ceil(
        std::max(0., boost::lexical_cast<double>(
                         get_unpack_arg("--bench-budget").asString()) /
                         time_per)));

    if (iters < 3) {
      std::cerr << "Note that only " << iters
                << " iter(s) can be run with the current budget" << std::endl;
    }

    double total_time = 0.f;
    double total_sqr_time = 0.f;

    bool ignore_accel = get_unpack_arg("--bench-ignore-accel").asBool();

    for (unsigned i = 0; i < iters; ++i) {
      Timer render_timer;
      double time_ignore_accel = b_render();
      double time = ignore_accel ? time_ignore_accel : render_timer.elapsed();
      total_time += time;
      total_sqr_time += time * time;
    }

    double mean_time = total_time / iters;
    double var_time = total_sqr_time / iters - mean_time * mean_time;
    double std_dev = std::sqrt(var_time);
    double std_error = std_dev / std::sqrt(iters);

    std::cout << "mean: " << mean_time << ", 2 * SE: " << 2 * std_error
              << std::endl;

  } else {
    render(true);
  }
  if (is_output_per_step) {
    VectorT<VectorT<FloatRGB>> step_outputs(n_output_steps,
                                            VectorT<FloatRGB>{width * height});
    VectorT<Span<FloatRGB>> outputs(step_outputs.begin(), step_outputs.end());
    render_gen(true, {tag_v<render::OutputType::OutputPerStep>, outputs});
    for (unsigned i = 0; i < step_outputs.size(); ++i) {
      QImage image(width, height, QImage::Format_RGB32);
      std::transform(
          step_outputs[i].begin(), step_outputs[i].end(),
          reinterpret_cast<BGRA32 *>(image.bits()),
          [&](const FloatRGB &v) { return float_rgb_to_bgra_32(v); });
      std::stringstream name_s;
      name_s << "out_" << i << ".png";
      image.save(name_s.str().c_str());
    }

    QImage summed_image(width, height, QImage::Format_RGB32);
    SpanSized<BGRA32> vals{reinterpret_cast<BGRA32 *>(image.bits()),
                           width * height};
    for (unsigned i = 0; i < vals.size(); ++i) {
      FloatRGB total = FloatRGB::Zero();

      for (unsigned j = 0; j < step_outputs.size(); ++j) {
        total += step_outputs[j][i];
      }

      vals[i] = float_rgb_to_bgra_32(total);
    }
    image.save("summed_image.png");
  }

  image.save(output_file_name.c_str());

  return 0;
}
