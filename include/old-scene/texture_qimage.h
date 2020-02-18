#include "scene/texture.h"

#include <QImage>
#include <thrust/optional.h>

namespace scene {
inline thrust::optional<TextureImage>
load_qimage(const std::string &file_name) {
  auto image = QImage(QString::fromStdString(file_name));
  if (image.isNull()) {
    return thrust::nullopt;
  }
  auto image_converted = image.convertToFormat(QImage::Format_RGB32);

  TextureImage tex;
  tex.width = static_cast<size_t>(image.width());
  tex.height = static_cast<size_t>(image.height());
  auto bytes = reinterpret_cast<const uint8_t *>(image.bits());
  size_t pixels = static_cast<size_t>(image.byteCount()) / 4;
  tex.data.resize(pixels);
  for (size_t i = 0; i < pixels; i++) {
    tex.data[i] =
        scene::Color(bytes[4 * i], bytes[4 * i + 1], bytes[4 * i + 2]) / 255.0f;
  }

  return tex;
}
} // namespace scene
