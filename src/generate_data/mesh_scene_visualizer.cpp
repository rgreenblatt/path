#include "generate_data/mesh_scene_generator.h"
#include "rng/uniform/uniform.h"

int main() {
  using namespace generate_data;
  MeshSceneGenerator generator;
  std::mt19937 rng(8);
  generator.generate(rng);
}
