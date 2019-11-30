#pragma once

#include "lib/unified_memory_vector.h"
#include "scene/color.h"

namespace scene {
struct TextureImageRef {
  unsigned width;
  unsigned height;
  const Color *data;

  inline HOST_DEVICE const Color &index(unsigned x, unsigned y) const {
    return data[x + y * width];
  }

  TextureImageRef(unsigned width, unsigned height, const Color *data)
      : width(width), height(height), data(data) {}
};

struct TextureImage {
  unsigned width;
  unsigned height;
  ManangedMemVec<Color> data;

#if 0
  TextureImage(const QImage &image);
#endif

  inline const Color &index(unsigned x, unsigned y) const {
    return data[x + y * width];
  }

  inline Color &index(unsigned x, unsigned y) { return data[x + y * width]; }

  inline TextureImageRef to_ref() const {
    return TextureImageRef(width, height, data.data());
  }
};

struct TextureData {
  unsigned index;
  float repeat_u;
  float repeat_v;

  TextureData(unsigned index, float repeat_u, float repeat_v)
      : index(index), repeat_u(repeat_u), repeat_v(repeat_v) {}

  inline const HOST_DEVICE Color &sample(const TextureImageRef *textures,
                                         const Eigen::Array2f &uv) const {
    const auto &texture = textures[index];
    unsigned x = static_cast<unsigned>(
                     uv[0] * static_cast<float>(texture.width) * repeat_u) %
                 (texture.width - 1);
    unsigned y = static_cast<unsigned>(
                     uv[1] * static_cast<float>(texture.height) * repeat_v) %
                 (texture.height - 1);

    return texture.index(x, y);
  }
};
} // namespace scene
