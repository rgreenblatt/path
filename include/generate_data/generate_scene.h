#pragma once

#include "generate_data/scene_triangles.h"
#include "scene/scene.h"

namespace generate_data {
scene::Scene generate_scene(const SceneTriangles &triangles);
}
