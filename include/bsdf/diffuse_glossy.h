#pragma once

#include "bsdf/combined.h"
#include "bsdf/diffuse.h"
#include "bsdf/glossy.h"

namespace bsdf {
using DiffuseGlossy = Combined<Diffuse, Glossy>;
}
