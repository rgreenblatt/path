#pragma once

#include "generate_data/triangle.h"
#include "generate_data/triangle_subset.h"

#include <iomanip>
#include <iostream>

namespace generate_data {
static void print_region(const TriangleSubset &subset) {
  subset.visit_tagged([&](auto tag, const auto &value) {
    if constexpr (tag == TriangleSubsetType::All) {
      std::cout << "'all'";
    } else if constexpr (tag == TriangleSubsetType::None) {
      std::cout << "None";
    } else {
      static_assert(tag == TriangleSubsetType::Some);
      std::cout << "[\n";
      for (const auto &p : value.outer()) {
        std::cout << "[" << p.x() << ", " << p.y() << "],\n";
      }
      std::cout << "]";
    }
    std::cout << "\n";
  });
}

static void print_multi_region(const TriangleMultiSubset &subset) {
  subset.visit_tagged([&](auto tag, const auto &value) {
    if constexpr (tag == TriangleSubsetType::All) {
      std::cout << "'all'";
    } else if constexpr (tag == TriangleSubsetType::None) {
      std::cout << "None";
    } else {
      static_assert(tag == TriangleSubsetType::Some);
      std::cout << "[\n";
      for (const auto &poly : value) {
        std::cout << "[\n";
        for (const auto &p : poly.outer()) {
          std::cout << "[" << p.x() << ", " << p.y() << "],\n";
        }
        std::cout << "],\n";
      }
      std::cout << "]";
    }
    std::cout << "\n";
  });
}
} // namespace generate_data
