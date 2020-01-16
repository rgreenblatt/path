#include "lib/span.h"
#include "lib/span_convertable_vector.h"

int main(int argc, char *argv[]) {
  int a = 0;
  SpanSized<int> s_size(&a, 0);

  // comment out below line and next line after this fails :)
  auto s_s = s_size;

  Span<int> s = s_size;
  Span<const int> s_const = s_size;
  SpanSized<const int> s_size_const = s_size;

  return 0;
}
