#include "generate_data/full_scene/scene_generator.h"

#include "intersect/triangle.h"
#include "lib/assert.h"
#include "lib/unit_vector.h"
#include "lib/vector_type.h"
#include "rng/uniform/uniform.h"
#include "scene/material.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <Eigen/Dense>
#include <tiny_obj_loader.h>

#include <iostream>
#include <optional>
#include <random>
#include <vector>

#include "dbg.h"

namespace generate_data {
namespace full_scene {
static VectorT<TriangleNormals> load_obj(const std::string &path) {
  VectorT<TriangleNormals> out;

  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> mesh_materials;

  std::string err;
  bool ret = tinyobj::LoadObj(&attrib, &shapes, &mesh_materials, &err,
                              path.c_str(), nullptr, true);
  if (!ret) {
    std::cerr << "Failed to load/parse .obj file" << std::endl;
    std::cerr << err << std::endl;
    unreachable();
  }

  for (size_t s = 0; s < shapes.size(); s++) {
    size_t index_offset = 0;
    for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
      unsigned fv = shapes[s].mesh.num_face_vertices[f];
      if (fv != 3) {
        std::cerr << "only triangles supported and obj contains non-triangles"
                  << std::endl;
        unreachable();
      }

      std::array<Eigen::Vector3f, 3> vertices;
      std::array<std::optional<UnitVector>, 3> normals_op;
      for (size_t v = 0; v < fv; v++) {
        tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
        tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
        tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
        tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];
        tinyobj::real_t nx;
        tinyobj::real_t ny;
        tinyobj::real_t nz;
        // TODO: currently texture not supported...
        tinyobj::real_t tx;
        tinyobj::real_t ty;

        if (idx.normal_index != -1) {
          nx = attrib.normals[3 * idx.normal_index + 0];
          ny = attrib.normals[3 * idx.normal_index + 1];
          nz = attrib.normals[3 * idx.normal_index + 2];
          normals_op[v] = UnitVector::new_normalize({nx, ny, nz});
        }

        if (idx.texcoord_index != -1) {
          tx = attrib.texcoords[2 * idx.texcoord_index + 0];
          ty = attrib.texcoords[2 * idx.texcoord_index + 1];
        } else {
          tx = 0;
          ty = 0;
        }

        Eigen::Vector3f vertex(vx, vy, vz);

        vertices[v] = vertex;
      }

      intersect::Triangle triangle{vertices};

      std::array<UnitVector, 3> normals;
      for (size_t v = 0; v < fv; v++) {
        normals[v] = optional_unwrap_or_else(normals_op[v],
                                             [&] { return triangle.normal(); });
      }

      out.push_back(TriangleNormals{
          .tri = triangle,
          .normals = normals,
      });

      index_offset += fv;
    }
  }

  return out;
}

SceneGenerator::SceneGenerator() {
  sphere_ = load_obj(OBJ_DIR_PATH "sphere.obj");
  monkey_ = load_obj(OBJ_DIR_PATH "monkey.obj");
  torus_ = load_obj(OBJ_DIR_PATH "torus.obj");
  meshs_ = {&sphere_, &torus_};
}

scene::Material random_bsdf(std::mt19937 &rng, bool force_emissive) {
  auto random_float_rgb = [&](auto dist) -> FloatRGB {
    return {{dist(rng), dist(rng), dist(rng)}};
  };

  auto random_weights = [&]() -> FloatRGB {
    std::uniform_real_distribution dist(0.f, 1.f);
    FloatRGB out = random_float_rgb(dist);
    auto mag = dist(rng);
    return out().matrix().normalized().array() * mag;
  };

  bool is_emissive = force_emissive || std::bernoulli_distribution(0.3)(rng);
  auto emission = [&]() -> FloatRGB {
    if (is_emissive) {
      return random_float_rgb(std::uniform_real_distribution(0.f, 70.f));
    } else {
      return FloatRGB::Zero();
    }
  }();

  auto bsdf = [&]() -> bsdf::UnionBSDF {
    bool is_only_transparent = std::bernoulli_distribution(0.2)(rng);

    // we could also have transparent + diffuse + glossy, but that't not
    // really needed atm
    if (is_only_transparent) {
      float ior = std::uniform_real_distribution(1.1f, 2.f)(rng);
      return {
          {tag_v<bsdf::BSDFType::DielectricRefractive>, random_weights(), ior}};
    }

    bool is_only_mirror = std::bernoulli_distribution(0.1)(rng);
    if (is_only_mirror) {
      return {{tag_v<bsdf::BSDFType::Mirror>, random_weights()}};
    }

    bool is_glossy = std::bernoulli_distribution(0.4)(rng);
    auto specular = random_weights();
    auto shininess = std::uniform_real_distribution(2.f, 50.f)(rng);
    bool is_diffuse = std::bernoulli_distribution(0.8)(rng);
    auto diffuse = random_weights();

    if (is_glossy && !is_diffuse) {
      return {{tag_v<bsdf::BSDFType::Glossy>, specular, shininess}};
    } else if (!is_glossy && is_diffuse) {
      return {{tag_v<bsdf::BSDFType::Diffuse>, diffuse}};
    } else {
      float div_factor = (diffuse + specular).maxCoeff();
      diffuse = diffuse / div_factor;
      specular = specular / div_factor;
      float total_diffuse_mass = diffuse.sum();
      float total_specular_mass = specular.sum();
      float diffuse_weight =
          total_diffuse_mass / (total_diffuse_mass + total_specular_mass);
      float glossy_weight = 1. - diffuse_weight;

      debug_assert((diffuse + specular).maxCoeff() < 1.f + 1e-4f);

      const auto new_diffuse = diffuse / diffuse_weight;
      const auto new_specular = specular / glossy_weight;

      return {{tag_v<bsdf::BSDFType::DiffuseGlossy>,
               {{{new_diffuse}, {new_specular, shininess}},
                {diffuse_weight, glossy_weight}}}};
    }
  }();

  return {
      .bsdf = bsdf,
      .emission = emission,
  };
}

// could actually use meshes part of scene_...
void SceneGenerator::add_mesh(const VectorT<TriangleNormals> &tris,
                              const Eigen::Affine3f &transform,
                              unsigned material_idx) {
  unsigned start_idx = scene_.triangles_.size();
  intersect::accel::AABB bounds;
  for (const auto &tri : tris) {
    std::array<UnitVector, 3> normals;
    auto new_tri = tri.tri.transform(transform);
    bounds = bounds.union_other(new_tri.bounds());
    scene_.triangles_.push_back_all(
        new_tri,
        scene::TriangleData{tri.normals, material_idx}.transform(transform));
  }
  unsigned end_idx = scene_.triangles_.size();

  bool is_emissive =
      (scene_.materials_[material_idx].emission().array() > 0.f).any();

  if (is_emissive) {
    scene_.emissive_clusters_.push_back({
        .material_idx = material_idx,
        .start_idx = start_idx,
        .end_idx = end_idx,
        .aabb = bounds,
    });
  }

  overall_aabb_ = overall_aabb_.union_other(bounds);
}

std::tuple<const scene::Scene &, unsigned>
SceneGenerator::generate(std::mt19937 &rng) {
  // clear! (could be more efficient...)
  scene_ = scene::Scene{};
  overall_aabb_ = intersect::accel::AABB::empty();

  unsigned mesh_count, individual_tri_count, total_size;
  while (true) {
    mesh_count = 0; // std::uniform_int_distribution(0u, 3u)(rng);
    individual_tri_count = std::uniform_int_distribution(0u, 30u)(rng);
    total_size = mesh_count + individual_tri_count;

    if (total_size >= 2) {
      break; // some interaction
    }
  }

  bool has_emissive = false;
  for (unsigned i = 0; i < total_size; ++i) {
    const auto material =
        random_bsdf(rng, !has_emissive && i == total_size - 1);
    has_emissive = has_emissive || material.emission.sum() > 1e-4f;
    scene_.materials_.push_back(material);
  }

  auto random_vec = [&](auto dist) -> Eigen::Vector3f {
    return {dist(rng), dist(rng), dist(rng)};
  };

  unsigned total_mesh_size = 0;
  for (unsigned i = 0; i < mesh_count; ++i) {
    std::uniform_real_distribution<float> angle_dist{-M_PI, M_PI};
    Eigen::Affine3f transform{
        Eigen::Translation3f{
            random_vec(std::uniform_real_distribution{-2.f, 2.f})} *
        Eigen::Scaling(std::uniform_real_distribution{0.1f, 1.f}(rng)) *
        Eigen::AngleAxisf{angle_dist(rng), Eigen::Vector3f::UnitX()} *
        Eigen::AngleAxisf{angle_dist(rng), Eigen::Vector3f::UnitY()} *
        Eigen::AngleAxisf{angle_dist(rng), Eigen::Vector3f::UnitZ()}};

    unsigned mesh_idx =
        std::uniform_int_distribution{size_t(0), meshs_.size() - 1}(rng);
    add_mesh(*meshs_[mesh_idx], transform, i);
    total_mesh_size += meshs_[mesh_idx]->size();
  }

  auto random_vert = [&]() {
    return random_vec(std::uniform_real_distribution{-0.5f, 0.5f});
  };

  for (unsigned j = mesh_count; j < total_size; ++j) {
    const auto center = random_vec(std::uniform_real_distribution{-2.f, 2.f});
    const float scale = std::bernoulli_distribution(0.2)(rng) ? 4.f : 1.f;
    const intersect::Triangle tri{
        scale * random_vert() + center,
        scale * random_vert() + center,
        scale * random_vert() + center,
    };
    const auto normal = tri.normal();

    add_mesh({TriangleNormals{.tri = tri, .normals = {normal, normal, normal}}},
             Eigen::Affine3f::Identity(), j);
  }

  // one mesh (everything is flattened anyway...)
  scene_.meshs_.push_back_all(scene_.triangles_.size(), overall_aabb_, "unique",
                              scene_.emissive_clusters_.size());
  scene_.transformed_objects_.push_back_all(
      intersect::TransformedObject(Eigen::Affine3f::Identity(), overall_aabb_),
      0);

  return {scene_, total_mesh_size};
}
} // namespace full_scene
} // namespace generate_data
