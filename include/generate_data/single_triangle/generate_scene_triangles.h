#pragma once

#include "generate_data/single_triangle/scene_triangles.h"
#include "rng/uniform/uniform.h"

namespace generate_data {
namespace single_triangle {
using UniformState = rng::uniform::Uniform<ExecutionModel::CPU>::Ref::State;
SceneTriangles generate_scene_triangles(UniformState &rng);
} // namespace single_triangle
} // namespace generate_data
