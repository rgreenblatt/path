#include "scene/scene.h"
#include "scene/camera.h"
#include "lib/group.h"
#include "scene/CS123XmlSceneParser.h"

#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <QFileInfo>
#include <QString>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <iostream>

namespace scene {
thrust::optional<Scene> Scene::make_scene(const std::string &filename,
                                          float width_height_ratio) {
  CS123XmlSceneParser parser(filename);
  if (!parser.parse()) {
    return thrust::nullopt;
  }
  CS123SceneCameraData cameraData;
  parser.get_camera_data(cameraData);

  Scene scene;

  scene.film_to_world_ = get_camera_transform(
      cameraData.look.head<3>(), cameraData.up.head<3>(),
      cameraData.pos.head<3>(), cameraData.heightAngle, width_height_ratio);

  CS123SceneGlobalData globalData;
  parser.get_global_data(globalData);
  scene.global_data_ = globalData;

  CS123SceneLightData light_data;
  for (int i = 0, size = parser.get_num_lights(); i < size; ++i) {
    parser.get_light_data(i, light_data);
    scene.add_light(light_data);
  }

  QFileInfo info(filename.c_str());
  std::string dir = info.path().toStdString();
  if (!scene.parse_tree(*parser.get_root_node(), dir + "/")) {
    return thrust::nullopt;
  }

  return scene;
}

bool Scene::parse_tree(const CS123SceneNode &root,
                       const std::string &base_dir) {
  return parse_node(root, Eigen::Affine3f::Identity(), base_dir) &&
         mesh_ends_.size() != 0;
}

bool Scene::parse_node(const CS123SceneNode &node,
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
    if (!add_primitive(*prim, transform, base_dir)) {
      return false;
    }
  }

  for (CS123SceneNode *child : node.children) {
    if (!parse_node(*child, transform, base_dir)) {
      return false;
    }
  }

  return true;
}

bool Scene::add_primitive(const CS123ScenePrimitive &prim,
                          const Eigen::Affine3f &transform,
                          const std::string &base_dir) {
  switch (prim.type) {
  case PrimitiveType::Mesh:
    return load_mesh(prim.meshfile, transform, base_dir);
  default:
    std::cerr << "We don't handle any other formats yet" << std::endl;
    return false;
  }
}

bool Scene::load_mesh(std::string file_path, const Eigen::Affine3f &transform,
                      const std::string &base_dir) {
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> mesh_materials;

  QFileInfo info((base_dir + file_path).c_str());

  auto absolute_path = info.absoluteFilePath().toStdString();

  auto map_it = loaded_meshes_.find(absolute_path);
  if (map_it != loaded_meshes_.end()) {
    mesh_instances_.push_back({map_it->second, transform});

    return true;
  }

  unsigned materials_offset = materials_.size();

  std::string err;
  bool ret = tinyobj::LoadObj(
      &attrib, &shapes, &mesh_materials, &err, absolute_path.c_str(),
      (info.absolutePath().toStdString() + "/").c_str(), true);
  if (!err.empty()) {
    std::cerr << err << std::endl;
    return false;
  }

  materials_.insert(materials_.end(), mesh_materials.begin(),
                    mesh_materials.end());

  if (!ret) {
    std::cerr << "Failed to load/parse .obj file" << std::endl;
    return false;
  }

  unsigned mesh_idx = mesh_ends_.size();

  Eigen::Vector3f min_b(std::numeric_limits<float>::max(),
                        std::numeric_limits<float>::max(),
                        std::numeric_limits<float>::max());
  Eigen::Vector3f max_b(std::numeric_limits<float>::lowest(),
                        std::numeric_limits<float>::lowest(),
                        std::numeric_limits<float>::lowest());

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
      std::array<Eigen::Vector3f, 3> colors;
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

        tinyobj::real_t red = attrib.colors[3 * idx.vertex_index + 0];
        tinyobj::real_t green = attrib.colors[3 * idx.vertex_index + 1];
        tinyobj::real_t blue = attrib.colors[3 * idx.vertex_index + 2];

        // TODO: UV
        Eigen::Vector3f vertex(vx, vy, vz);
        min_b = min_b.cwiseMin(vertex);
        max_b = max_b.cwiseMax(vertex);
        vertices[v] = vertex;
        normals[v] = Eigen::Vector3f(nx, ny, nz).normalized();
        colors[v] = Eigen::Vector3f(red, green, blue);
      }

      unsigned material_idx = shapes[s].mesh.material_ids[f] + materials_offset;

      triangles_.push_back(Triangle{vertices});
      triangle_data_.push_back(TriangleData{normals, material_idx, colors});

      index_offset += fv;
    }
  }

  mesh_ends_.push_back(triangles_.size());
  mesh_aabbs_.push_back({min_b, max_b});
  mesh_instances_.push_back({mesh_idx, transform});
  loaded_meshes_.insert({absolute_path, mesh_idx});

  return true;
}

void Scene::add_light(const CS123SceneLightData &data) {
  lights_.push_back(data);
}
} // namespace scene
