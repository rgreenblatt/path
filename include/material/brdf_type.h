#pragma once

namespace material {
enum class BRDFType {
  Diffuse,
  Glossy,
  Mirror,
  DielectricRefractive,
};

template <BRDFType type> class BRDF;
}; // namespace material
