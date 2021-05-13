#pragma once

#include "render/settings.h"

#include <filesystem>

namespace render {
Settings load_config(const std::filesystem::path &config_file_path);

void print_config(const Settings &settings);
} // namespace render
