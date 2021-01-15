#include "lib/bgra_32.h"
#include "lib/float_rgb.h"
#include "lib/serialize_eigen.h"
#include "lib/span.h"

#include <QImage>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>

#include <fstream>

int main() {
  std::vector<FloatRGB> float_rgb;
  {
    std::ifstream file("benchmark_ground_truth/cornell_diffuse.bin",
                       std::ios::binary);
    cereal::BinaryInputArchive ar(file);
    ar(float_rgb);
  }

  unsigned width = static_cast<unsigned>(std::sqrt(float_rgb.size()));

  QImage image(width, width, QImage::Format_RGB32);
  Span<BGRA32> pixels(reinterpret_cast<BGRA32 *>(image.bits()), width * width);

  for (unsigned i = 0; i < width * width; ++i) {
    pixels[i] = float_rgb_to_bgra_32(float_rgb[i]);
  }

  image.save("loaded_out.png");
}
