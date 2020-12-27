#include "lib/info/timer.h"
#include "render/renderer_from_files.h"

#include <benchmark/benchmark.h>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>

#include <string>
#include <limits>
#include <unordered_set>
#include <filesystem>
#include <fstream>

#include <Eigen/Core>

namespace cereal {
template<typename Archive>
void serialize(Archive& ar, Eigen::Array3f& arr) {
  ar(arr[0], arr[1], arr[2]);
}
}

int main(int argc, char *argv[]) {
  namespace fs = std::filesystem;

  struct BenchItem {
    std::string name;
    fs::path scene_path;
    unsigned n_truth_samples;
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

  assert(names.size() == bench_items.size());
  assert(paths.size() == bench_items.size());

  unsigned max_width = std::numeric_limits<unsigned>::lowest();
  for (unsigned width : desired_widths) {
    max_width = std::max(max_width, width);
  }

  render::RendererFromFiles renderer;
  renderer.load_config("configs/benchmark.yaml");

  const fs::path ground_truth_save_directory =
      "benchmark_ground_truth";
  const std::string ext = ".bin";

  std::vector<std::vector<Eigen::Array3f>> intensities(bench_items.size());

  for (unsigned i = 0; i < bench_items.size(); ++i) {
    const auto &[name, scene_path, n_truth_samples, max_size] = bench_items[i];
    auto ground_truth_file = ground_truth_save_directory / (name + ext);

    bool successfully_loaded = false;
    if (fs::exists(ground_truth_file)) {
      std::ifstream file(ground_truth_file);
      cereal::BinaryInputArchive ar(file);
      ar(intensities[i]);
      successfully_loaded = intensities[i].size() == max_width * max_width;
    }

    if (!successfully_loaded) {
      std::ofstream file(ground_truth_file);
      cereal::BinaryOutputArchive ar(file);

      intensities[i].resize(max_width * max_width);

      renderer.load_scene(scene_path, 1.f);
      unsigned n_truth_samples_in = n_truth_samples;
      renderer.render_intensities(ExecutionModel::GPU, intensities[i],
                                  n_truth_samples_in, max_width, max_width);
      if (n_truth_samples_in != n_truth_samples)  {
        std::cerr << "n_truth_samples had to be changed: invalid" <<
          std::endl;
        assert(false);
        abort();
      }

      ar(intensities[i]);
    }
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
}
