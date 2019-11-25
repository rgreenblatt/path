#include "ray/render.h"
#include "scene/cs123_scene.h"

#include <QImage>
#include <boost/lexical_cast.hpp>

#include <chrono>
#include <iostream>

template <ray::ExecutionModel execution_model>
void run_test(unsigned width, unsigned height, const scene::Scene &scene,
              const Eigen::Affine3f &transform, const std::string &filename,
              unsigned depth, unsigned x_special, unsigned y_special) {

  QImage image(width, height, QImage::Format_RGB32);

  auto bgra_data = reinterpret_cast<BGRA *>(image.bits());

  ray::Renderer<execution_model> renderer(width, height, depth, x_special,
                                          y_special);

#if 0
  // realistic memory benchmark
  renderer.render(scene, bgra_data, static_cast<scene::Transform>(transform));
#endif

  auto start = std::chrono::high_resolution_clock::now();
  renderer.render(scene, bgra_data, static_cast<scene::Transform>(transform));
  std::cout << "rendered in "
            << std::chrono::duration_cast<std::chrono::duration<double>>(
                   std::chrono::high_resolution_clock::now() - start)
                   .count()
            << std::endl;

  image.save(filename.c_str());
}

int main(int argc, char *argv[]) {
  if (argc <= 4) {
    std::cout << "too few args" << std::endl;

    return 1;
  }

  const unsigned depth = boost::lexical_cast<unsigned>(argv[2]);
  const unsigned x_special = boost::lexical_cast<unsigned>(argv[3]);
  const unsigned y_special = boost::lexical_cast<unsigned>(argv[4]);

  constexpr unsigned width = 1230;
  constexpr unsigned height = 778;

  for (size_t i = 1; i < /* static_cast<size_t>(argc) */ 2; i++) {
    scene::CS123Scene scene(argv[i], width, height);

    const std::string file_name =
        "out_" + boost::lexical_cast<std::string>(i) + ".png";

    std::cout << "rendering cpu" << std::endl;
    run_test<ray::ExecutionModel::CPU>(width, height, scene, scene.transform(),
                                       "cpu_" + file_name, depth, x_special,
                                       y_special);
    std::cout << "=============" << std::endl;
    std::cout << "rendering gpu" << std::endl;
    run_test<ray::ExecutionModel::GPU>(width, height, scene, scene.transform(),
                                       "gpu_" + file_name, depth, x_special,
                                       y_special);
  }

  return 0;
}
