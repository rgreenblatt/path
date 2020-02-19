#include <chrono>
#include <iostream>

class Timer {
public:
  Timer() { start = std::chrono::high_resolution_clock::now(); }

  void report(const std::string &name) {
    std::cout << name << ": "
              << std::chrono::duration_cast<std::chrono::duration<double>>(
                     std::chrono::high_resolution_clock::now() - start)
                     .count()
              << std::endl;
  }

private:
  std::chrono::high_resolution_clock::time_point start;
};
