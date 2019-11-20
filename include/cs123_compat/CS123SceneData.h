/**
 * @file CS123SceneData.h
 *
 * Header file containing scene data structures.
 */

#pragma once

#include <string>
#include <vector>

#include "scene/material.h"

enum class LightType { Point, Directional, Spot, Area };

enum class PrimitiveType { Cube, Cone, Cylinder, Torus, Sphere, Mesh };

// Enumeration for types of transformations that can be applied to objects,
// lights, and cameras.
enum class TransformationType { Translate, Scale, Rotate, Matrix };

template <typename Enumeration>
auto as_integer(Enumeration const value) ->
    typename std::underlying_type<Enumeration>::type {
  return static_cast<typename std::underlying_type<Enumeration>::type>(value);
}

// Struct to store a BGRA color in floats [0,1]
using CS123SceneColor = scene::Color;

// Scene global color coefficients
struct CS123SceneGlobalData {
  float ka; // global ambient coefficient
  float kd; // global diffuse coefficient
  float ks; // global specular coefficient
  float kt; // global transparency coefficient
};

// Data for a single light
struct CS123SceneLightData {
  int id;
  LightType type;

  CS123SceneColor color;
  Eigen::Array3f function; // Attenuation function

  Eigen::Vector3f pos; // Not applicable to directional lights
  Eigen::Vector3f dir; // Not applicable to point lights

  float radius;   // Only applicable to spot lights
  float penumbra; // Only applicable to spot lights
  float angle;    // Only applicable to spot lights

  float width, height; // Only applicable to area lights
};

// Data for scene camera
struct CS123SceneCameraData {
  Eigen::Vector3f pos;
  Eigen::Vector3f look;
  Eigen::Vector3f up;

  float heightAngle;
  float aspectRatio;

  float aperture;    // Only applicable for depth of field
  float focalLength; // Only applicable for depth of field
};

// Data for file maps (ie: texture maps)
struct CS123SceneFileMap {
  //    CS123SceneFileMap() : texid(0) {}
  bool isUsed;
  std::string filename;
  float repeatU;
  float repeatV;

  void clear() {
    isUsed = false;
    repeatU = 0.0f;
    repeatV = 0.0f;
    filename = std::string();
  }
};

// Data for scene materials
struct CS123SceneMaterial {
  // This field specifies the diffuse color of the object. This is the color you
  // need to use for the object in sceneview. You can get away with ignoring the
  // other color values until intersect and ray.
  //   CS123SceneMaterial() {}
  CS123SceneColor cDiffuse;

  CS123SceneColor cAmbient;
  CS123SceneColor cReflective;
  CS123SceneColor cSpecular;
  CS123SceneColor cTransparent;
  CS123SceneColor cEmissive;

  CS123SceneFileMap textureMap;
  float blend;

  CS123SceneFileMap bumpMap;

  float shininess;

  float ior; // index of refraction

  void clear() {
    cAmbient[0] = 0.0f;
    cAmbient[1] = 0.0f;
    cAmbient[2] = 0.0f;
    cDiffuse[0] = 0.0f;
    cDiffuse[1] = 0.0f;
    cDiffuse[2] = 0.0f;
    cSpecular[0] = 0.0f;
    cSpecular[1] = 0.0f;
    cSpecular[2] = 0.0f;
    cReflective[0] = 0.0f;
    cReflective[1] = 0.0f;
    cReflective[2] = 0.0f;
    cTransparent[0] = 0.0f;
    cTransparent[1] = 0.0f;
    cTransparent[2] = 0.0f;
    cEmissive[0] = 0.0f;
    cEmissive[1] = 0.0f;
    cEmissive[2] = 0.0f;
    textureMap.clear();
    bumpMap.clear();
    blend = 0.0f;
    shininess = 0.0f;
    ior = 0.0;
  }
};

struct CS123ScenePrimitive {
  PrimitiveType type;
  std::string meshfile; // Only applicable to meshes
  CS123SceneMaterial material;
};

// Data for transforming a scene object. Aside from the TransformationType, the
// remaining of the data in the struct is mutually exclusive.
struct CS123SceneTransformation {
  TransformationType type;

  Eigen::Vector3f translate; // The translation vector. Only valid if
                             // transformation is a translation.
  Eigen::Vector3f
      scale; // The scale vector. Only valid if transformation is a scale.
  Eigen::Vector3f rotate; // The axis of rotation. Only valid if the
                          // transformation is a rotation.
  float angle; // The rotation angle in RADIANS. Only valid if transformation is
               // a rotation.

  Eigen::Matrix4f matrix; // The matrix for the transformation. Only valid if
                          // the transformation is a custom matrix.
};

// Structure for non-primitive scene objects
struct CS123SceneNode {
  std::vector<CS123SceneTransformation *> transformations;

  std::vector<CS123ScenePrimitive *> primitives;

  std::vector<CS123SceneNode *> children;
};
