#include "generate_data/single_triangle/generate_scene.h"

#include "scene/triangle_constructor.h"

namespace generate_data {
namespace single_triangle {
scene::Scene generate_scene(const SceneTriangles &triangles) {
  scene::Material onto_material{
      .bsdf = {{tag_v<bsdf::BSDFType::Diffuse>,
                {.diffuse = {{0.f, 0.f, 1.f}}}}},
      .emission = FloatRGB{{0.f, 0.f, 0.f}},
  };
  scene::Material blocking_material{
      .bsdf = {{tag_v<bsdf::BSDFType::Diffuse>,
                {.diffuse = {{0.f, 0.f, 0.f}}}}},
      .emission = FloatRGB{{0.f, 0.f, 0.f}},
  };
  scene::Material light_material{
      .bsdf = {{tag_v<bsdf::BSDFType::Diffuse>,
                {.diffuse = {{0.f, 0.f, 0.f}}}}},
      .emission = FloatRGB{{50.f, 50.f, 50.f}},
  };

  scene::TriangleConstructor scene_constructor;

  scene_constructor.add_triangle(triangles.triangle_onto.template cast<float>(),
                                 scene_constructor.add_material(onto_material));
  scene_constructor.add_triangle(
      triangles.triangle_blocking.template cast<float>(),
      scene_constructor.add_material(blocking_material));
  scene_constructor.add_triangle(
      triangles.triangle_light.template cast<float>(),
      scene_constructor.add_material(light_material));

  return scene_constructor.scene("generated_scene");
}
} // namespace single_triangle
} // namespace generate_data
