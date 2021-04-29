#pragma once

#include "render/mega_kernel_settings.h"
#include "render/streaming_settings.h"

namespace render {
// There are two overall approach to ray/path tracing used in this project:
//  - mega kernel
//  - streaming
//
// In the mega kernel approach, we will launch one kernel which does the entire
// initialize + intersect + shade and repeat pipeline. (we may also need one
// other kernel to "reduce" the outputs of this kernel depending on the exact
// parametters).
//
// In the streaming approach, we instead have separate kernels for
// initialization, intersection, and shading.
enum class KernelApproach {
  MegaKernel,
  Streaming, // also called "wavefront"
};

using KernelApproachSettings =
    TaggedUnion<KernelApproach, MegaKernelSettings, StreamingSettings>;

using KernelApproachCompileTime =
    TaggedUnion<KernelApproach, MegaKernelSettings::CompileTime,
                StreamingSettings::CompileTime>;

} // namespace render
