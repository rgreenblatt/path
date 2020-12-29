#include "lib/serialize_eigen.h"
#include "lib/bgra.h"
#include "lib/span.h"

#include <Eigen/Core>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <QImage>

#include <fstream>

int main() {
  std::vector<Eigen::Array3f> intensities;
  {
    std::ifstream file("benchmark_ground_truth/cornell_glass_sphere.bin",
                       std::ios::binary);
    cereal::BinaryInputArchive ar(file);
    ar(intensities);
  }

  unsigned width = static_cast<unsigned>(std::sqrt(intensities.size()));

  QImage image(width, width, QImage::Format_RGB32);
  Span<BGRA> pixels(reinterpret_cast<BGRA *>(image.bits()), width * width);

  for (unsigned i = 0; i < width * width; ++i) {
    pixels[i] = intensity_to_bgr(intensities[i]);
  }

  image.save("loaded_out.png");
}
