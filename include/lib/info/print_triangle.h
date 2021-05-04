#pragma once

#include "intersect/triangle.h"

void print_triangle(const intersect::Triangle &triangle) {
  std::cout << "[" << std::endl;
  for (const auto &vert : triangle.vertices) {
    std::cout << "[" << vert[0] << ", " << vert[1] << ", " << vert[2] << "],"
              << std::endl;
  }
  std::cout << "]\n" << std::endl;
}
