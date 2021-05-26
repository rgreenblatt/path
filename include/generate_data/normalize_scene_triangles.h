#pragma once

#include "generate_data/scene_triangles.h"

namespace generate_data {
// This is specific to this exact scene.
SceneTriangles normalize_scene_triangles(const SceneTriangles &tris);
} // namespace generate_data
