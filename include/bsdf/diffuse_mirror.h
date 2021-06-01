#pragma once

#include "bsdf/combined.h"
#include "bsdf/diffuse.h"
#include "bsdf/mirror.h"

namespace bsdf {
using DiffuseMirror = Combined<Diffuse, Mirror>;
}
