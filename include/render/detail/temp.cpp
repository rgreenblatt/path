#include "halton.h"

#include <iostream>

int main() {
  using namespace render::detail;
  std::cout << halton_sequence<2>[1][0] << std::endl;
  std::cout << halton_sequence<2>[1][1] << std::endl;
  std::cout << halton_sequence<2>[2][0] << std::endl;
  std::cout << halton_sequence<2>[2][1] << std::endl;

  std::cout << halton<2>(1)[0] << std::endl;
  std::cout << halton<2>(1)[1] << std::endl;
  std::cout << halton<2>(2)[0] << std::endl;
  std::cout << halton<2>(2)[1] << std::endl;
}
