#include "lib/assert.h"
#include "lib/info/timer.h"
#include "render/renderer_from_files.h"

#include <benchmark/benchmark.h>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>

#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_set>

#include <Eigen/Core>

namespace cereal {
template <typename Archive> void serialize(Archive &ar, Eigen::Array3f &arr) {
  ar(arr[0], arr[1], arr[2]);
}
} // namespace cereal

int main(int argc, char *argv[]) {
  namespace fs = std::filesystem;

  struct BenchItem {
    std::string name;
    fs::path scene_path;
    unsigned n_ground_truth_samples;
    unsigned max_size;
  };

  std::vector<unsigned> samples = {1, 4, 8, 16, 64, 256, 512, 2048, 8192};
  std::vector<unsigned> desired_widths = {8, 64, 128, 256, 1024};

  std::vector<BenchItem> bench_items = {
      {
          "cornell_diffuse",
          "scenes/cornell-diffuse.xml",
          65536,
          1 << 26,
      },
      {
          "cornell_glossy",
          "scenes/cornell-glossy.xml",
          65536,
          1 << 26,
      },
      {
          "cornell_glass_box",
          "scenes/cornell-glass-box.xml",
          65536,
          1 << 26,
      },
      {
          "cornell_glass_sphere",
          "scenes/cornell-glass_sphere.xml",
          65536,
          1 << 23,
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

  auto is_power_of_2 = [](unsigned n) { return n > 0 && ((n & (n - 1)) == 0); };

  for (unsigned width : desired_widths) {
    always_assert(ground_truth_width % width == 0);
    always_assert(is_power_of_2(ground_truth_width / width));
  }

  render::RendererFromFiles renderer;
  renderer.load_config("configs/benchmark.yaml");

  const fs::path ground_truth_save_directory = "benchmark_ground_truth";
  const std::string ext = ".bin";

  std::vector<std::vector<Eigen::Array3f>> intensities(bench_items.size());

  for (unsigned i = 0; i < bench_items.size(); ++i) {
    const auto &[name, scene_path, n_ground_truth_samples, max_size] =
        bench_items[i];
    auto ground_truth_file = ground_truth_save_directory / (name + ext);

    bool successfully_loaded = false;
    if (fs::exists(ground_truth_file)) {
      std::ifstream file(ground_truth_file);
      cereal::BinaryInputArchive ar(file);
      ar(intensities[i]);
      successfully_loaded =
          intensities[i].size() == ground_truth_width * ground_truth_width;
    }

    if (!successfully_loaded) {
      std::ofstream file(ground_truth_file);
      cereal::BinaryOutputArchive ar(file);

      intensities[i].resize(ground_truth_width * ground_truth_width);

      renderer.load_scene(scene_path, 1.f);
      unsigned n_truth_samples_in = n_ground_truth_samples;
      std::cout << "rendering " << scene_path << " for benchmark " << name
                << " with " << n_ground_truth_samples << " samples, "
                << ground_truth_width << "x" << ground_truth_width << std::endl;
      renderer.render_intensities(ExecutionModel::GPU, intensities[i],
                                  n_truth_samples_in, ground_truth_width,
                                  ground_truth_width, true);
      if (n_truth_samples_in != n_ground_truth_samples) {
        std::cerr << "n_ground_truth_samples had to be changed: invalid"
                  << std::endl;
        unreachable();
      }

      ar(intensities[i]);
    }
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
}
