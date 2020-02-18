#pragma once

#include "scene/CS123ISceneParser.h"
#include "scene/CS123SceneData.h"

#include <map>
#include <vector>

#include <QtXml>

namespace scene {
/**
 * @class CS123XmlSceneParser
 *
 * This class parses the scene graph specified by the CS123 Xml file format.
 *
 * The parser is designed to replace the TinyXML parser that was in turn
 * designed to replace the Flex/Yacc/Bison parser.
 */
class CS123XmlSceneParser : public CS123ISceneParser {

public:
  // Create a parser, passing it the scene file.
  CS123XmlSceneParser(const std::string &filename);

  // Clean up all data for the scene
  virtual ~CS123XmlSceneParser();

  // Parse the scene.  Returns false if scene is invalid.
  virtual bool parse();

  virtual bool get_global_data(CS123SceneGlobalData &data) const;

  virtual bool get_camera_data(CS123SceneCameraData &data) const;

  virtual CS123SceneNode *get_root_node() const;

  virtual int get_num_lights() const;

  // Returns the ith light data
  virtual bool get_light_data(int i, CS123SceneLightData &data) const;

private:
  // The filename should be contained within this parser implementation.
  // If you want to parse a new file, instantiate a different parser.
  bool parse_global_data(const QDomElement &globaldata);
  bool parse_camera_data(const QDomElement &cameradata);
  bool parse_light_data(const QDomElement &lightdata);
  bool parse_object_data(const QDomElement &object);
  bool parse_trans_block(const QDomElement &transblock, CS123SceneNode *node);
  bool parse_primitive(const QDomElement &prim, CS123SceneNode *node);

  std::string file_name;
  mutable std::map<std::string, CS123SceneNode *> objects_;
  CS123SceneCameraData camera_data_;
  std::vector<CS123SceneLightData *> lights_;
  CS123SceneGlobalData global_data_;
  std::vector<CS123SceneNode *> nodes_;
};
} // namespace scene
