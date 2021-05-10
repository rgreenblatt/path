#include "lib/bgra_32.h"
#include "lib/float_rgb.h"
#include "lib/float_rgb_image_utils.h"
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

  std::vector<FloatRGB> float_rgb(width * width);
  renderer.render(
      ExecutionModel::GPU, width, width,
      render::Output{tag_v<render::OutputType::FloatRGB>, float_rgb},

      samples, true);
  unsigned reduced_size = reduced_width * reduced_width;
  std::vector<FloatRGB> reduced_float_rgb(reduced_size);
  unsigned reduced_samples = samples * reduction_factor * reduction_factor;
  renderer.render(
      ExecutionModel::GPU, reduced_width, reduced_width,
      render::Output{tag_v<render::OutputType::FloatRGB>, reduced_float_rgb},
      reduced_samples, true);

  std::vector<FloatRGB> downsampled_float_rgb(reduced_size);
  downsample_to(float_rgb, downsampled_float_rgb, width, reduced_width);

  double mean_absolute_error = compute_mean_absolute_error(
      reduced_float_rgb, downsampled_float_rgb, reduced_width);
  std::cout << "mean_absolute_error: " << mean_absolute_error << std::endl;

  QImage reduced_image(reduced_width, reduced_width, QImage::Format_RGB32);
  QImage downsampled_image(reduced_width, reduced_width, QImage::Format_RGB32);
  QImage difference_image(reduced_width, reduced_width, QImage::Format_RGB32);

  Span<BGRA32> reduced_pixels(reinterpret_cast<BGRA32 *>(reduced_image.bits()),
                              reduced_size);
  Span<BGRA32> downsampled_pixels(
      reinterpret_cast<BGRA32 *>(downsampled_image.bits()), reduced_size);
  Span<BGRA32> difference_pixels(
      reinterpret_cast<BGRA32 *>(difference_image.bits()), reduced_size);

  for (unsigned i = 0; i < reduced_size; ++i) {
    reduced_pixels[i] = float_rgb_to_bgra_32(reduced_float_rgb[i]);
    downsampled_pixels[i] = float_rgb_to_bgra_32(downsampled_float_rgb[i]);
    difference_pixels[i] = float_rgb_to_bgra_32(
        (reduced_float_rgb[i] - downsampled_float_rgb[i]).abs() /
        mean_absolute_error);
  }

  reduced_image.save("reduced_image.png");
  downsampled_image.save("downsampled_image.png");
  // difference should look random
  difference_image.save("difference_image.png");
}
