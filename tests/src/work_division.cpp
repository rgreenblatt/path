#include "work_division/work_division.h"
#include "work_division/settings.h"
#include "work_division/work_division_impl.h"

#include <gtest/gtest.h>

using namespace work_division;

static unsigned division_samples_per(const WorkDivision &division) {
  return division.num_sample_blocks() * division.sample_block_size() *
             division.base_samples_per_thread() +
         division.n_threads_per_unit_extra();
}

static unsigned effective_x_dim(const WorkDivision &division) {
  return division.x_block_size() * division.num_x_blocks();
}

static unsigned effective_y_dim(const WorkDivision &division) {
  return division.y_block_size() * division.num_y_blocks();
}

// this is a bit gross and maybe somewhat implementation specific :(
static void check_coverage(const WorkDivision &division) {
  unsigned samples_per = division_samples_per(division);
  unsigned x_dim = effective_x_dim(division);
  unsigned y_dim = effective_y_dim(division);
  unsigned samples_covered_up_to = 0;

  unsigned x_covered_up_to = 0;
  unsigned y_covered_up_to = 0;

  for (unsigned block_idx = 0; block_idx < division.total_num_blocks();
       ++block_idx) {
    for (unsigned thread_idx = 0; thread_idx < division.block_size();
         ++thread_idx) {
      auto [start_sample, end_sample, x, y] =
          division.get_thread_info(block_idx, thread_idx);

      ASSERT_EQ(start_sample, samples_covered_up_to);
      ASSERT_EQ(x, x_covered_up_to);
      ASSERT_EQ(y, y_covered_up_to);

      samples_covered_up_to = end_sample;
      ASSERT_LE(samples_covered_up_to, samples_per);
      if (samples_covered_up_to == samples_per &&
          (thread_idx + 1) % division.sample_block_size() == 0) {
        ++x_covered_up_to;
        samples_covered_up_to = 0;
        if (x_covered_up_to % division.x_block_size() == 0) {
          if (thread_idx == division.block_size() - 1) {
            if (x_covered_up_to == x_dim) {
              ASSERT_EQ((block_idx + 1) % division.num_x_blocks(), 0);
              ++y_covered_up_to;
              x_covered_up_to = 0;
            } else {
              ASSERT_GE(y_covered_up_to, division.y_block_size() - 1);
              y_covered_up_to -= division.y_block_size() - 1;
            }
          } else {
            ++y_covered_up_to;
            ASSERT_GE(x_covered_up_to, division.x_block_size());
            x_covered_up_to -= division.x_block_size();
          }
        }
      }
      ASSERT_LE(x_covered_up_to, x_dim);
      ASSERT_LE(y_covered_up_to, y_dim);
    }
  }
  ASSERT_EQ(samples_covered_up_to, 0);
  ASSERT_EQ(x_covered_up_to, 0);
  ASSERT_EQ(y_covered_up_to, y_dim);
}

static void check_dims_as_expected(const WorkDivision &division,
                                   unsigned samples_per, unsigned x_dim,
                                   unsigned y_dim) {
  EXPECT_EQ(division_samples_per(division), samples_per);
  EXPECT_GE(effective_x_dim(division), x_dim);
  EXPECT_GE(effective_y_dim(division), y_dim);
  EXPECT_LT(effective_x_dim(division), x_dim + division.x_block_size());
  EXPECT_LT(effective_y_dim(division), y_dim + division.y_block_size());
}

TEST(WorkDivision, combination) {
  Settings settings{64, 16, 4, 8};

  for (unsigned x_dim : {1, 3, 17, 72, 128}) {
    for (unsigned y_dim : {1, 3, 17, 72, 128}) {
      for (unsigned samples_per : {1, 3, 17, 72, 128, 2048}) {
        WorkDivision division(settings, samples_per, x_dim, y_dim);
        check_dims_as_expected(division, samples_per, x_dim, y_dim);
        check_coverage(division);
      }
    }
  }
}

TEST(WorkDivision, default_settings) {
  Settings settings;

  WorkDivision division(settings, 2048, 8, 8);
  check_dims_as_expected(division, 2048, 8, 8);
  check_coverage(division);
}
