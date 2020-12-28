#include "lib/bgra.h"
#include "lib/intensity_utils.h"
#include "render/renderer_from_files.h"

#include <QImage>

#include <iostream>

int main() {
  render::RendererFromFiles renderer;
  renderer.load_config("configs/benchmark.yaml");
  renderer.load_scene("scenes/cornell-diffuse.xml", 1.f);

  unsigned width = 128;
  unsigned reduced_width = 32;
  unsigned reduction_factor = width / reduced_width;

  unsigned samples = 1024;

  std::vector<Eigen::Array3f> intensities(width * width);
  renderer.render_intensities(ExecutionModel::GPU, intensities, samples, width,
                              width, true);
  unsigned reduced_size = reduced_width * reduced_width;
  std::vector<Eigen::Array3f> reduced_intensities(reduced_size);
  unsigned reduced_samples = samples * reduction_factor * reduction_factor;
  renderer.render_intensities(ExecutionModel::GPU, reduced_intensities,
                              reduced_samples, reduced_width, reduced_width,
                              true);

  std::vector<Eigen::Array3f> downsampled_intensities(reduced_size);
  downsample_to(intensities, downsampled_intensities, width, reduced_width);

  double mean_absolute_error = compute_mean_absolute_error(
      reduced_intensities, downsampled_intensities, reduced_width);
  std::cout << "mean_absolute_error: " << mean_absolute_error << std::endl;

  QImage reduced_image(reduced_width, reduced_width, QImage::Format_RGB32);
  QImage downsampled_image(reduced_width, reduced_width, QImage::Format_RGB32);
  QImage difference_image(reduced_width, reduced_width, QImage::Format_RGB32);

  Span<BGRA> reduced_pixels(reinterpret_cast<BGRA *>(reduced_image.bits()),
                            reduced_size);
  Span<BGRA> downsampled_pixels(
      reinterpret_cast<BGRA *>(downsampled_image.bits()), reduced_size);
  Span<BGRA> difference_pixels(
      reinterpret_cast<BGRA *>(difference_image.bits()), reduced_size);

  for (unsigned i = 0; i < reduced_size; ++i) {
    reduced_pixels[i] = intensity_to_bgr(reduced_intensities[i]);
    downsampled_pixels[i] = intensity_to_bgr(downsampled_intensities[i]);
    difference_pixels[i] = intensity_to_bgr(
        (reduced_intensities[i] - downsampled_intensities[i]).abs() /
        mean_absolute_error);
  }

  reduced_image.save("reduced_image.png");
  downsampled_image.save("downsampled_image.png");
  // difference should look random
  difference_image.save("difference_image.png");
}
