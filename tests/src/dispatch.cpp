#include "meta/all_values/sequential_dispatch.h"

auto iden = [](auto tag) { return tag(); };
static_assert(sequential_dispatch<10>(0, iden) == 0);
static_assert(sequential_dispatch<10>(3, iden) == 3);
static_assert(sequential_dispatch<10>(9, iden) == 9);
static_assert(sequential_dispatch<10>(9, iden) == 9);
static_assert(sequential_dispatch<31>(0, iden) == 0);
static_assert(sequential_dispatch<31>(30, iden) == 30);
static_assert(sequential_dispatch<32>(0, iden) == 0);
static_assert(sequential_dispatch<32>(30, iden) == 30);
static_assert(sequential_dispatch<32>(31, iden) == 31);
static_assert(sequential_dispatch<33>(0, iden) == 0);
static_assert(sequential_dispatch<33>(31, iden) == 31);
static_assert(sequential_dispatch<33>(32, iden) == 32);
static_assert(sequential_dispatch<148>(98, iden) == 98);
