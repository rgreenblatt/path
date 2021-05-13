#include "render/config_io.h"

#include "lib/assert.h"
#include "lib/serialize_enum.h"
#include "lib/serialize_tagged_union.h"
#include "meta/as_tuple/serialize.h"

#include <cereal-yaml/archives/yaml.hpp>

#include <fstream>
#include <iostream>

namespace render {
Settings load_config(const std::filesystem::path &config_file_path) {
  std::ifstream i(config_file_path);
  always_assert(!i.fail());
  always_assert(!i.bad());
  always_assert(i.is_open());
  cereal::YAMLInputArchive archive(i);

  Settings settings;
  archive(cereal::make_nvp("settings", settings));

  return settings;
}

void print_config(const Settings &settings) {
  std::ostringstream os;
  {
    cereal::YAMLOutputArchive archive(os);
    archive(cereal::make_nvp("settings", settings));
  }
  std::cout << os.str() << std::endl;
}
} // namespace render
