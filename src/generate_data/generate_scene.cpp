#include "generate_data/generate_scene.h"
#include "scene/triangle_constructor.h"

namespace generate_data {
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

  scene_constructor.add_triangle(triangles.triangle_onto,
                                 scene_constructor.add_material(onto_material));
  scene_constructor.add_triangle(
      triangles.triangle_blocking,
      scene_constructor.add_material(blocking_material));
  scene_constructor.add_triangle(
      triangles.triangle_light, scene_constructor.add_material(light_material));

  return scene_constructor.scene("generated_scene");
}
} // namespace generate_data
