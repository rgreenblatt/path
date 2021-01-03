#pragma once

// This header is custom - honestly pretty gross...
#include <cereal/name_value_pair.hpp>

// convenience macro for use in serialize
#define NVP(T) CEREAL_NVP(T)
