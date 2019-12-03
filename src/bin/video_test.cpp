#include "ray/render.h"
#include "scene/camera.h"
#include "scene/pool_scene.h"

#include "ProgressBar.hpp"
#include <QImage>
#include <boost/lexical_cast.hpp>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <dbg.h>
#include <iostream>

template <ray::ExecutionModel execution_model>
void render_frames(unsigned width, unsigned height,
                   unsigned super_sampling_rate, scene::PoolScene &scene,
                   const Eigen::Affine3f &transform,
                   const std::string &dir_name, unsigned depth,
                   bool use_kd_tree, unsigned frames, float frame_rate,
                   unsigned physics_super_sampling_rate) {
#if 0
  if (!std::filesystem::exists(dir_name)) {
    std::filesystem::create_directories(dir_name);
  }
#endif

  ray::Renderer<execution_model> renderer(width, height, super_sampling_rate,
                                          depth);

  std::cout << "start:" << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  float secs_per_frame = 1.0f / (frame_rate * physics_super_sampling_rate);

  ProgressBar progress(frames, 60);

  cv::VideoWriter writer(dir_name + "out.avi", CV_FOURCC('M', 'J', 'P', 'G'),
                         frame_rate, cv::Size(width, height));

  cv::Mat frame(height, width, CV_8UC4);
  cv::Mat out(height, width, CV_8UC3);

  for (unsigned i = 0; i < frames; i++) {
    auto bgra_data = reinterpret_cast<BGRA *>(frame.data);

    renderer.render(scene, bgra_data, static_cast<scene::Transform>(transform),
                    use_kd_tree, false);

    for (unsigned i = 0; i < physics_super_sampling_rate; i++) {
      scene.step(secs_per_frame);
    }

    cv::cvtColor(frame, out, CV_RGBA2RGB);
    writer.write(out);

    ++progress;
    progress.display();
  }

  double time = std::chrono::duration_cast<std::chrono::duration<double>>(
                    std::chrono::high_resolution_clock::now() - start)
                    .count();

  std::cout << std::endl;
  std::cout << "rendered video in " << time << std::endl;
  std::cout << frames / time << " fps" << std::endl;
  std::cout << std::endl;

  writer.release();
}

int main(int argc, char *argv[]) {
  if (argc != 10) {
    std::cout << "wrong num args" << std::endl;

    return 1;
  }

  const unsigned depth = boost::lexical_cast<unsigned>(argv[1]);
  const unsigned width = boost::lexical_cast<unsigned>(argv[2]);
  const unsigned height = boost::lexical_cast<unsigned>(argv[3]);
  const unsigned super_sampling_rate = boost::lexical_cast<unsigned>(argv[4]);
  const bool use_kd_tree = boost::lexical_cast<bool>(argv[5]);
  const bool render_cpu = boost::lexical_cast<bool>(argv[6]);
  const unsigned frames = boost::lexical_cast<unsigned>(argv[7]);
  const float frame_rate = boost::lexical_cast<float>(argv[8]);
  const unsigned physics_super_sampling_rate =
      boost::lexical_cast<unsigned>(argv[9]);

  scene::PoolScene pool_scene;

  const std::string file_name = "out.png";

  auto transform = scene::get_camera_transform(
      Eigen::Vector3f(-1, -1, -1), Eigen::Vector3f(0, 1, 0),
      Eigen::Vector3f(20, 20, 20), 1.0f, width, height);

  if (render_cpu) {
    std::cout << "rendering cpu" << std::endl;
    render_frames<ray::ExecutionModel::CPU>(
        width, height, super_sampling_rate, pool_scene, transform, "cpu_video/",
        depth, use_kd_tree, frames, frame_rate, physics_super_sampling_rate);
    std::cout << "=============" << std::endl;
  }

  std::cout << "rendering gpu" << std::endl;
  render_frames<ray::ExecutionModel::GPU>(
      width, height, super_sampling_rate, pool_scene, transform, "gpu_video/",
      depth, use_kd_tree, frames, frame_rate, physics_super_sampling_rate);

  return 0;
}
