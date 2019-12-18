#pragma once

#include "scene/scene.h"
#include "lib/span.h"

namespace ray {
namespace detail {
class SceneRef {
  SceneRef(scene::ShapeData *shapes, unsigned num_shapes, scene::Light *lights,
           unsigned num_lights)
      : shapes_(shapes, num_shapes), lights_(lights, num_lights) {}


private:
  Span<scene::ShapeData> shapes_;
  Span<scene::Light> lights_;
};

} // namespace detail
} // namespace ray
