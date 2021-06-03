#pragma once

#include "generate_data/single_triangle/scene_triangles.h"
#include "scene/scene.h"

namespace generate_data {
namespace single_triangle {
scene::Scene generate_scene(const SceneTriangles &triangles);
}
} // namespace generate_data
