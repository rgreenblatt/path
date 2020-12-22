#pragma once

#include <chrono>
#include <iostream>

class Timer {
public:
  Timer() { start = std::chrono::high_resolution_clock::now(); }

  double elapsed() {
    return std::chrono::duration_cast<std::chrono::duration<double>>(
               std::chrono::high_resolution_clock::now() - start)
        .count();
  }

  void report(const std::string &name) {
    std::cout << name << ": " << elapsed() << std::endl;
  }

private:
  std::chrono::high_resolution_clock::time_point start;
};
