#include "lib/bit_utils.h"

static_assert(popcount(0b1100u) == 2);
static_assert(popcount(0b0u) == 0);
static_assert(popcount(0b11111110u) == 7);

static_assert(count_leading_zeros(static_cast<uint8_t>(0b1u)) == 7);
static_assert(count_leading_zeros(static_cast<uint16_t>(0b1u)) == 15);
static_assert(count_leading_zeros(static_cast<uint16_t>(0b1111u)) == 12);
static_assert(count_leading_zeros(0b1u) == 31);
static_assert(count_leading_zeros(static_cast<uint64_t>(0b1u)) == 63);
static_assert(count_leading_zeros(0b100u) == 29);
static_assert(count_leading_zeros(0b10111u) == 27);
static_assert(count_leading_zeros(std::numeric_limits<unsigned>::max()) == 0);

static_assert(log_2_floor(0b1u) == 0);
static_assert(log_2_floor(2u) == 1);
static_assert(log_2_floor(8u) == 3);
static_assert(log_2_floor(15u) == 3);
static_assert(log_2_floor(32u) == 5);
static_assert(log_2_floor(63u) == 5);

static_assert(!power_of_2(0b0u));
static_assert(power_of_2(0b1u));
static_assert(power_of_2(0b10u));
static_assert(power_of_2(0b100000u));
static_assert(!power_of_2(0b100100u));
static_assert(!power_of_2(0b111111u));
static_assert(!power_of_2(0b111u));
static_assert(!power_of_2(0b101u));
static_assert(!power_of_2(std::numeric_limits<unsigned>::max()));

static_assert(closest_power_of_2(0b100u) == 0b100u);
static_assert(closest_power_of_2(0b0u) == 0b1u);
static_assert(closest_power_of_2(0b101u) == 0b100u);
static_assert(closest_power_of_2(0b111u) == 0b1000u);
static_assert(closest_power_of_2(std::numeric_limits<unsigned>::max()) ==
              1u << 31);

static_assert(bit_mask<unsigned>(0) == 0b1);
static_assert(bit_mask<unsigned>(2) == 0b100);
static_assert(bit_mask<unsigned>(7) == 0b10000000);
static_assert(bit_mask<unsigned>(31) == 0b10000000000000000000000000000000u);

static_assert(up_to_mask<unsigned>(0) == 0b1);
static_assert(up_to_mask<unsigned>(1) == 0b11);
static_assert(up_to_mask<unsigned>(7) == 0b11111111);
static_assert(up_to_mask<unsigned>(31) == 0b11111111111111111111111111111111u);
