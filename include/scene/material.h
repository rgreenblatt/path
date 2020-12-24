#pragma once

#include "bsdf/material.h"
#include "bsdf/union_bsdf.h"

namespace scene {
// for now, only this material is supported as a scene material
using Material = bsdf::Material<bsdf::UnionBSDF>;
} // namespace scene
