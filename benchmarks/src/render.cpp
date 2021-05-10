#include "lib/assert.h"
#include "lib/float_rgb_image_utils.h"
#include "lib/info/timer.h"
#include "lib/serialize_eigen.h"
#include "render/renderer_from_files.h"

#include <benchmark/benchmark.h>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_set>

int main(int argc, char *argv[]) {
  namespace fs = std::filesystem;

  struct BenchItem {
    std::string name;
    fs::path scene_path;
    unsigned n_ground_truth_samples;
    unsigned max_size;
  };

  std::vector<unsigned> samples = {1,  4,  8,   12,  16,  24,  32,  48,
                                   64, 96, 128, 192, 256, 512, 2048};
  std::vector<unsigned> desired_widths = {8, 64, 128, 256, 1024};

  std::vector<BenchItem> bench_items = {
      {
          "cornell_diffuse",
          "scenes/cornell-diffuse.xml",
          8192,
          1 << 23,
      },
      {
          "cornell_glossy",
          "scenes/cornell-glossy.xml",
          8192,
          1 << 23,
      },
      {
          "cornell_glass_box",
          "scenes/cornell-glass-box.xml",
          8192,
          1 << 22,
      },
      {
          "cornell_glass_sphere",
          "scenes/cornell-glass-sphere.xml",
          4096,
          1 << 19,
      },
  };

  std::unordered_set<std::string> names;
  std::unordered_set<std::string> paths;
  for (const auto &[name, scene_path, a, b] : bench_items) {
    names.insert(name);
    paths.insert(scene_path);
  }

  always_assert(names.size() == bench_items.size());
  always_assert(paths.size() == bench_items.size());

  unsigned max_width = std::numeric_limits<unsigned>::lowest();
  for (unsigned width : desired_widths) {
    max_width = std::max(max_width, width);
  }
  unsigned ground_truth_width = max_width;

  for (unsigned width : desired_widths) {
    always_assert(ground_truth_width % width == 0);
  }

  render::RendererFromFiles renderer;
  renderer.load_config("configs/benchmark.yaml");
  renderer.print_config();

  const fs::path ground_truth_save_directory = "benchmark_ground_truth";
  const std::string ext = ".bin";

  std::vector<std::vector<FloatRGB>> ground_truth_float_rgb(bench_items.size());

  fs::create_directories(ground_truth_save_directory);

  for (unsigned i = 0; i < bench_items.size(); ++i) {
    const auto &[name, scene_path, n_ground_truth_samples, max_size] =
        bench_items[i];
    auto ground_truth_file = ground_truth_save_directory / (name + ext);

    bool successfully_loaded = false;
    if (fs::exists(ground_truth_file)) {
      {
        std::ifstream file(ground_truth_file, std::ios::binary);
        cereal::BinaryInputArchive ar(file);
        ar(ground_truth_float_rgb[i]);
      }
      successfully_loaded = ground_truth_float_rgb[i].size() ==
                            ground_truth_width * ground_truth_width;
      if (!successfully_loaded) {
        std::cout << "loaded file has differnent than expected size"
                  << std::endl;
      }
    }

    if (!successfully_loaded) {
      std::ofstream file(ground_truth_file, std::ios::binary);
      cereal::BinaryOutputArchive ar(file);

      std::cout << "rendering " << scene_path << " with "
                << n_ground_truth_samples << " samples, " << ground_truth_width
                << "x" << ground_truth_width << std::endl;

      renderer.load_scene(scene_path, 1.f);

      ground_truth_float_rgb[i].resize(ground_truth_width * ground_truth_width);

      renderer.render(ExecutionModel::GPU, ground_truth_width,
                      ground_truth_width,
                      render::Output{tag_v<render::OutputType::FloatRGB>,
                                     ground_truth_float_rgb[i]},
                      n_ground_truth_samples, true);

      ar(ground_truth_float_rgb[i]);
    }
  }

  std::vector<std::vector<std::vector<FloatRGB>>>
      resized_ground_truth_float_rgb(bench_items.size());

  // loop over references so lambda capture is valid...
  for (unsigned i = 0; i < bench_items.size(); ++i) {
    const auto &bench_item = bench_items[i];
    const auto &name = bench_item.name;
    const auto &scene_path = bench_item.scene_path;
    // scene_path, n_ground_truth_samples, max_size
    resized_ground_truth_float_rgb[i].resize(desired_widths.size());
    for (unsigned j = 0; j < desired_widths.size(); ++j) {
      const unsigned &desired_width = desired_widths[j];
      auto &ground_truth = resized_ground_truth_float_rgb[i][j];
      ground_truth.resize(desired_width * desired_width);
      downsample_to(ground_truth_float_rgb[i], ground_truth, ground_truth_width,
                    desired_width);
      for (const unsigned &n_samples : samples) {
        uint64_t total_size =
            static_cast<uint64_t>(n_samples) * desired_width * desired_width;
        // odd compare needed to avoid overflow
        if (total_size > bench_item.max_size) {
          continue;
        }
        std::stringstream ss;
        ss << name << "_" << n_samples << "_" << desired_width << "x"
           << desired_width;
        uint64_t base_num_iters = 5;
        uint64_t max_num_iters = 100;
        unsigned iterations =
            std::min(base_num_iters * bench_item.max_size / total_size,
                     static_cast<uint64_t>(max_num_iters));
        benchmark::RegisterBenchmark(
            ss.str().c_str(),
            [&](benchmark::State &st) {
              renderer.load_scene(scene_path, 1.f, true);
              std::vector<FloatRGB> bench_float_rgb(desired_width *
                                                    desired_width);
              auto render = [&] {
                unsigned actual_n_samples = n_samples;
                renderer.render(
                    ExecutionModel::GPU, desired_width, desired_width,
                    render::Output{tag_v<render::OutputType::FloatRGB>,
                                   bench_float_rgb},
                    actual_n_samples, false);
                always_assert(actual_n_samples == n_samples);
              };

              // warmup (maybe not needed)
              render();

              // right now there isn't any way to seed render, so the error
              // is constant (at least it should be aside from parallelism
              // related non-determinism due to reduction)
              for (auto _ : st) {
                render();
              }

              st.counters["error"] = compute_mean_absolute_error(
                  bench_float_rgb, ground_truth, desired_width);
            })
            ->Unit(benchmark::kMillisecond)
            ->Iterations(iterations);
      }
    }
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
}
