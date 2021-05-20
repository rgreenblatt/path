#include "generate_data/gen_data.h"

int main() {
  generate_data::gen_data(1, 1, 1, 20);
  generate_data::deinit_renderers();
}
