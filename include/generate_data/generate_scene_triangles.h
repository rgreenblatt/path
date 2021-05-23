#pragma once

#include "generate_data/scene_triangles.h"
#include "rng/uniform/uniform.h"

namespace generate_data {
using UniformState = rng::uniform::Uniform<ExecutionModel::CPU>::Ref::State;
SceneTriangles generate_scene_triangles(UniformState &rng);
} // namespace generate_data
