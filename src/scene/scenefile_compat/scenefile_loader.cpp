#include "scene/scenefile_compat/scenefile_loader.h"
#include "lib/group.h"
#include "lib/utils.h"
#include "scene/camera.h"
#include "scene/mat_to_material.h"
#include "scene/scene.h"
#include "scene/scenefile_compat/CS123XmlSceneParser.h"

#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <QFileInfo>
#include <QString>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <iostream>

namespace scene {
namespace scenefile_compat {
thrust::optional<Scene> ScenefileLoader::load_scene(const std::string &filename,
                                                    float width_height_ratio) {
  loaded_meshes_.clear();

  overall_min_b_transformed_ = max_eigen_vec();
  overall_max_b_transformed_ = lowest_eigen_vec();

  CS123XmlSceneParser parser(filename);
  if (!parser.parse()) {
    return thrust::nullopt;
  }
  CS123SceneCameraData cameraData;
  parser.get_camera_data(cameraData);

  Scene scene_v;

  scene_v.film_to_world_ = get_camera_transform(
      cameraData.look.head<3>(), cameraData.up.head<3>(),
      cameraData.pos.head<3>(), cameraData.heightAngle, width_height_ratio);

  CS123SceneGlobalData globalData;
  parser.get_global_data(globalData);
  global_data_ = globalData; // TODO: Should global data be used?

  CS123SceneLightData light_data;
  for (int i = 0, size = parser.get_num_lights(); i < size; ++i) {
    parser.get_light_data(i, light_data);
    if (!add_light(scene_v, light_data)) {
      return thrust::nullopt;
    }
  }

  QFileInfo info(filename.c_str());
  std::string dir = info.path().toStdString();
  if (!parse_tree(scene_v, *parser.get_root_node(), dir + "/")) {
    return thrust::nullopt;
  }

  scene_v.overall_aabb_ = {overall_min_b_transformed_,
                           overall_max_b_transformed_};

  return scene_v;
}

bool ScenefileLoader::parse_tree(Scene &scene_v, const CS123SceneNode &root,
                                 const std::string &base_dir) {
  return parse_node(scene_v, root, Eigen::Affine3f::Identity(), base_dir) &&
         scene_v.mesh_ends_.size() != 0;
}

bool ScenefileLoader::parse_node(Scene &scene_v, const CS123SceneNode &node,
                                 const Eigen::Affine3f &parent_transform,
                                 const std::string &base_dir) {
  Eigen::Affine3f transform = parent_transform;
  for (CS123SceneTransformation *trans : node.transformations) {
    switch (trans->type) {
    case TransformationType::Translate:
      transform = transform * Eigen::Translation<float, 3>(trans->translate);
      break;
    case TransformationType::Scale:
      transform = transform * Eigen::Scaling(trans->scale);
      break;
    case TransformationType::Rotate:
      transform =
          transform * Eigen::AngleAxis<float>(trans->angle, trans->rotate);
      break;
    case TransformationType::Matrix:
      transform = transform * trans->matrix;
      break;
    }
  }

  for (CS123ScenePrimitive *prim : node.primitives) {
    if (!add_primitive(scene_v, *prim, transform, base_dir)) {
      return false;
    }
  }

  for (CS123SceneNode *child : node.children) {
    if (!parse_node(scene_v, *child, transform, base_dir)) {
      return false;
    }
  }

  return true;
}

bool ScenefileLoader::add_primitive(Scene &scene_v,
                                    const CS123ScenePrimitive &prim,
                                    const Eigen::Affine3f &transform,
                                    const std::string &base_dir) {
  switch (prim.type) {
  case PrimitiveType::Mesh:
    return load_mesh(scene_v, prim.meshfile, transform, base_dir);
  default:
    std::cerr << "We don't handle any other formats yet" << std::endl;
    return false;
  }
}

bool ScenefileLoader::load_mesh(Scene &scene_v, std::string file_path,
                                const Eigen::Affine3f &transform,
                                const std::string &base_dir) {
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> mesh_materials;

  QFileInfo info((base_dir + file_path).c_str());

  auto absolute_path = info.absoluteFilePath().toStdString();

  auto add_mesh_instance = [&](unsigned idx,
                               const intersect::accel::AABB &aabb) {
    auto transformed_aabb = aabb.transform(transform);
    scene_v.transformed_objects_.push_back({idx, transform, transformed_aabb});
    overall_max_b_transformed_ =
        overall_max_b_transformed_.cwiseMax(transformed_aabb.get_max_bound());
    overall_min_b_transformed_ =
        overall_min_b_transformed_.cwiseMin(transformed_aabb.get_min_bound());
  };

  auto map_it = loaded_meshes_.find(absolute_path);
  if (map_it != loaded_meshes_.end()) {
    add_mesh_instance(map_it->second, scene_v.mesh_aabbs_[map_it->second]);

    return true;
  }

  unsigned materials_offset = scene_v.materials_.size();

  std::string err;
  bool ret = tinyobj::LoadObj(
      &attrib, &shapes, &mesh_materials, &err, absolute_path.c_str(),
      (info.absolutePath().toStdString() + "/").c_str(), true);
  if (!err.empty()) {
    std::cerr << err << std::endl;
    return false;
  }

  for (const auto &m : mesh_materials) {
    auto material = mat_to_material(m);

    is_emissive.push_back(material.emission().matrix().squaredNorm() > 1e-9f);

    scene_v.materials_.push_back(material);
  }

  if (!ret) {
    std::cerr << "Failed to load/parse .obj file" << std::endl;
    return false;
  }

  unsigned mesh_idx = scene_v.mesh_ends_.size();

  auto min_b = max_eigen_vec();
  auto max_b = lowest_eigen_vec();

  Eigen::Vector3f emissive_group_min_b;
  Eigen::Vector3f emissive_group_max_b;
  bool adding_to_emissive_group = false;
  unsigned emissive_material_idx;
  unsigned emissive_start_idx;

  auto end_emissive_group = [&] {
    unsigned emissive_end_idx = scene_v.triangles_.size();
    scene_v.emissive_groups_.push_back(
        {emissive_material_idx,
         emissive_start_idx,
         emissive_end_idx,
         {emissive_group_min_b, emissive_group_max_b}});
  };

  // TODO populate vectors and use tranform
  for (size_t s = 0; s < shapes.size(); s++) {
    size_t index_offset = 0;
    for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
      unsigned fv = shapes[s].mesh.num_face_vertices[f];
      if (fv != 3) {
        std::cerr << "only triangles supported and obj contains non-triangles"
                  << std::endl;
        return false;
      }

      std::array<Eigen::Vector3f, 3> vertices;
      std::array<Eigen::Vector3f, 3> normals;
      for (size_t v = 0; v < fv; v++) {
        tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
        tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
        tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
        tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];
        tinyobj::real_t nx;
        tinyobj::real_t ny;
        tinyobj::real_t nz;
        tinyobj::real_t tx;
        tinyobj::real_t ty;

        if (idx.normal_index != -1) {
          nx = attrib.normals[3 * idx.normal_index + 0];
          ny = attrib.normals[3 * idx.normal_index + 1];
          nz = attrib.normals[3 * idx.normal_index + 2];
        } else {
          nx = 0;
          ny = 0;
          nz = 0;
        }
        if (idx.texcoord_index != -1) {
          tx = attrib.texcoords[2 * idx.texcoord_index + 0];
          ty = attrib.texcoords[2 * idx.texcoord_index + 1];
        } else {
          tx = 0;
          ty = 0;
        }

        Eigen::Vector3f vertex(vx, vy, vz);
        min_b = min_b.cwiseMin(vertex);
        max_b = max_b.cwiseMax(vertex);

        vertices[v] = vertex;
        normals[v] = Eigen::Vector3f(nx, ny, nz).normalized();
      }

      unsigned material_idx = shapes[s].mesh.material_ids[f] + materials_offset;
      if (shapes[s].mesh.material_ids[f] == -1) {

        std::cerr << "IDXS -1!!!" << std::endl;
      }

      auto add_vertices_emissive_group = [&] {
        for (const auto &vertex : vertices) {
          emissive_group_min_b = emissive_group_min_b.cwiseMin(vertex);
          emissive_group_max_b = emissive_group_max_b.cwiseMax(vertex);
        }
      };

      auto new_emissive_group = [&] {
        emissive_group_min_b = max_eigen_vec();
        emissive_group_max_b = lowest_eigen_vec();
        adding_to_emissive_group = true;
        emissive_start_idx = scene_v.triangles_.size();
        emissive_material_idx = material_idx;

        add_vertices_emissive_group();
      };

      if (is_emissive[material_idx]) {
        if (adding_to_emissive_group) {
          if (material_idx != emissive_material_idx) {
            // end, new group
            end_emissive_group();

            new_emissive_group();
          } else {
            add_vertices_emissive_group();
          }
        } else {
          new_emissive_group();
        }
      } else if (adding_to_emissive_group) {
        end_emissive_group();

        adding_to_emissive_group = false;
      }

      // No normals..
      if (normals[0].squaredNorm() < 1e-5) {
        auto normal =
            ((vertices[1] - vertices[0]).cross(vertices[2] - vertices[0]))
                .normalized()
                .eval();
        for (auto &n : normals) {
          n = normal;
        }
      }

      scene_v.triangles_.push_back({vertices});
      scene_v.triangle_data_.push_back({normals, material_idx});

      index_offset += fv;
    }
  }

  if (adding_to_emissive_group) {
    end_emissive_group();
  }

  scene_v.emissive_group_ends_per_mesh_.push_back(
      scene_v.emissive_groups_.size());
  scene_v.mesh_ends_.push_back(scene_v.triangles_.size());
  intersect::accel::AABB aabb{min_b, max_b};
  scene_v.mesh_aabbs_.push_back(aabb);
  add_mesh_instance(mesh_idx, aabb);
  loaded_meshes_.insert({absolute_path, mesh_idx});
  scene_v.mesh_paths_.push_back(absolute_path);

  std::cout << "added mesh" << std::endl;
  std::cout << "total triangle count: " << scene_v.triangles_.size()
            << std::endl;
  std::cout << "total num emissive groups: " << scene_v.emissive_groups_.size()
            << std::endl;

  return true;
}

bool ScenefileLoader::add_light(Scene &, const CS123SceneLightData &) {
  std::cerr << "point/directional lights not currently supported" << std::endl;
  return false;
}
} // namespace scenefile_compat
} // namespace scene
