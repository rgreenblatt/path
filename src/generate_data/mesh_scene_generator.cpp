#include "generate_data/mesh_scene_generator.h"

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

MeshSceneGenerator::MeshSceneGenerator() {
  sphere_ = load_obj("scenes/models/sphere.obj");
  monkey_ = load_obj("scenes/models/monkey.obj");
  torus_ = load_obj("scenes/models/torus.obj");
}

bsdf::UnionBSDF random_bsdf(std::mt19937 &rng) {
  auto random_float_rgb = [&](auto dist) -> FloatRGB {
    return {{dist(rng), dist(rng), dist(rng)}};
  };

  auto random_weights = [&]() -> FloatRGB {
    std::uniform_real_distribution dist(0.f, 1.f);
    FloatRGB out = random_float_rgb(dist);
    auto mag = dist(rng);
    return out().matrix().normalized().array() * mag;
  };

  bool is_emissive = std::bernoulli_distribution(0.3)(rng);
  auto emission = [&]() -> FloatRGB {
    if (is_emissive) {
      return random_float_rgb(std::uniform_real_distribution(0.f, 70.f));
    } else {
      return FloatRGB::Zero();
    }
  }();

#if 0
  auto bsdf = [&]() -> bsdf::UnionBSDF {
    bool is_only_transparent = std::bernoulli_distribution(0.3)(rng);

    // we could also have transparent + diffuse + glossy, but that't not
    // really needed atm
    if (is_only_transparent) {
      float ior = std::uniform_real_distribution(1.2f, 2.f)(rng);
      return {
          {tag_v<bsdf::BSDFType::DielectricRefractive>, random_weights(), ior}};
    }

    bool is_only_mirror = std::bernoulli_distribution(0.3)(rng);
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
      return {
          {tag_v<bsdf::BSDFType::GlossyDiffuse>, diffuse, specular, shininess}};
    }
  }();
#endif
}

void MeshSceneGenerator::generate(std::mt19937 &rng) {
  while (true) {
    unsigned mesh_count = std::uniform_int_distribution(0u, 3u)(rng);
    unsigned individual_tri_count = std::uniform_int_distribution(0u, 15u)(rng);

    // if (mesh_count == 0 && individual_tri_count == 0);
  }
}

// template <> void MeshSceneGenerator::generate() {
} // namespace generate_data
