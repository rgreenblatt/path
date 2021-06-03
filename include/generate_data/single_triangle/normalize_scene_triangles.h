#pragma once

#include "generate_data/single_triangle/scene_triangles.h"

namespace generate_data {
namespace single_triangle {
// This is specific to this exact scene.
SceneTriangles normalize_scene_triangles(const SceneTriangles &tris);
} // namespace single_triangle
} // namespace generate_data
