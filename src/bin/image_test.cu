#include "ray/render.h"
#include "scene/cs123_scene.h"

#include <QImage>
#include <boost/lexical_cast.hpp>

#include <chrono>
#include <iostream>

template <ray::ExecutionModel execution_model>
void run_test(unsigned width, unsigned height, unsigned super_sampling_rate,
              const scene::Scene &scene, const Eigen::Affine3f &transform,
              const std::string &filename, unsigned depth, bool use_kd_tree) {

  QImage image(width, height, QImage::Format_RGB32);

  auto bgra_data = reinterpret_cast<BGRA *>(image.bits());

  ray::Renderer<execution_model> renderer(width, height, super_sampling_rate,
                                          depth);

#if 1
  // realistic memory benchmark
  renderer.render(scene, bgra_data, static_cast<scene::Transform>(transform),
                  use_kd_tree);
#endif

  std::cout << "start:" << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
  renderer.render(scene, bgra_data, static_cast<scene::Transform>(transform),
                  use_kd_tree);
  std::cout << "rendered in "
            << std::chrono::duration_cast<std::chrono::duration<double>>(
                   std::chrono::high_resolution_clock::now() - start)
                   .count()
            << std::endl;

  image.save(filename.c_str());
}

int main(int argc, char *argv[]) {
  if (argc != 8) {
    std::cout << "wrong num args" << std::endl;

    return 1;
  }

  const unsigned depth = boost::lexical_cast<unsigned>(argv[2]);
  const unsigned width = boost::lexical_cast<unsigned>(argv[3]);
  const unsigned height = boost::lexical_cast<unsigned>(argv[4]);
  const unsigned super_sampling_rate = boost::lexical_cast<unsigned>(argv[5]);
  const bool use_kd_tree = boost::lexical_cast<bool>(argv[6]);
  const bool render_cpu = boost::lexical_cast<bool>(argv[7]);

  for (size_t i = 1; i < /* static_cast<size_t>(argc) */ 2; i++) {
    scene::CS123Scene scene(argv[i], width, height);

    const std::string file_name =
        "out_" + boost::lexical_cast<std::string>(i) + ".png";

    if (render_cpu) {
      std::cout << "rendering cpu" << std::endl;
      run_test<ray::ExecutionModel::CPU>(
          width, height, super_sampling_rate, scene, scene.transform(),
          "cpu_" + file_name, depth, use_kd_tree);
      std::cout << "=============" << std::endl;
    }

    std::cout << "rendering gpu" << std::endl;
    run_test<ray::ExecutionModel::GPU>(
        width, height, super_sampling_rate, scene, scene.transform(),
        "gpu_" + file_name, depth, use_kd_tree);
  }

  return 0;
}
