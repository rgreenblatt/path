#include "lib/optional.h"
#include "meta/mock.h"

#include <cassert>
#include <gtest/gtest.h>

TEST(Optional, basic) {
  Optional<unsigned> a;
  EXPECT_FALSE(a.has_value());
  a = nullopt_value;
  EXPECT_FALSE(a.has_value());
  Optional<unsigned> b = nullopt_value;
  EXPECT_FALSE(b.has_value());
  Optional<unsigned> c = 8;
  ASSERT_TRUE(c.has_value());
  EXPECT_EQ(*c, 8);
  c = 2;
  ASSERT_TRUE(c.has_value());
  EXPECT_EQ(*c, 2);
}

template <bool copy_allowed> class CountCalls {
public:
  CountCalls() = delete;
  
  CountCalls(unsigned *moved_into, unsigned *assign_moved,
             unsigned *copied_into, unsigned *assign_copied, bool *destructed)
      : moved_into_(moved_into), assign_moved_(assign_moved),
        copied_into_(copied_into), assign_copied_(assign_copied),
        destructed_(destructed) {}


  CountCalls(const CountCalls &other) requires copy_allowed {
    assert(other.destructed_ != nullptr);
    assert(!*other.destructed_);
    moved_into_ = other.moved_into_;
    assign_moved_ = other.assign_moved_;
    copied_into_ = other.copied_into_;
    assign_copied_ = other.assign_copied_;
    destructed_ = other.destructed_;
    ++*copied_into_;
  }

  CountCalls(CountCalls &&other) {
    assert(other.destructed_ != nullptr);
    assert(!*other.destructed_);
    moved_into_ = other.moved_into_;
    assign_moved_ = other.assign_moved_;
    copied_into_ = other.copied_into_;
    assign_copied_ = other.assign_copied_;
    destructed_ = other.destructed_;
    other.moved_into_ = nullptr;
    other.assign_moved_ = nullptr;
    other.copied_into_ = nullptr;
    other.assign_copied_ = nullptr;
    // *other.destructed_ = true;
    other.destructed_ = nullptr;
    ++*moved_into_;
  }

  CountCalls &operator=(const CountCalls &other) requires copy_allowed {
    assert(destructed_ != nullptr);
    assert(other.destructed_ != nullptr);
    assert(!*destructed_);
    assert(!*other.destructed_);
    if (this != &other) {
      moved_into_ = other.moved_into_;
      assign_moved_ = other.assign_moved_;
      copied_into_ = other.copied_into_;
      assign_copied_ = other.assign_copied_;
      destructed_ = other.destructed_;
      ++*assign_copied_;
    }
    return *this;
  };

  CountCalls &operator=(CountCalls &&other) {
    assert(other.destructed_ != nullptr);
    assert(!*destructed_);
    assert(!*other.destructed_);
    if (this != &other) {
      moved_into_ = other.moved_into_;
      assign_moved_ = other.assign_moved_;
      copied_into_ = other.copied_into_;
      assign_copied_ = other.assign_copied_;
      destructed_ = other.destructed_;
      other.moved_into_ = nullptr;
      other.assign_moved_ = nullptr;
      other.copied_into_ = nullptr;
      other.assign_copied_ = nullptr;
      // *other.destructed_ = true;
      other.destructed_ = nullptr;
      ++*assign_moved_;
    }
    return *this;
  }

  ~CountCalls() {
    if (destructed_ != nullptr) {
      assert(!*destructed_);
      *destructed_ = true;
    }
  }

private:
  unsigned *moved_into_;
  unsigned *assign_moved_;
  unsigned *copied_into_;
  unsigned *assign_copied_;
  bool *destructed_;
};

TEST(Optional, MoveDestruct) {
  {
    Optional<CountCalls<false>> a;
    EXPECT_FALSE(a.has_value());
  }
  {
    Optional<CountCalls<false>> a = nullopt_value;
    EXPECT_FALSE(a.has_value());
  }
  {
    Optional<CountCalls<false>> a;
    EXPECT_FALSE(a.has_value());
    a = nullopt_value;
  }

  unsigned moved_into;
  unsigned assign_moved;
  unsigned copied_into;
  unsigned assign_copied;
  bool destructed;

  auto reset = [&] {
    moved_into = 0;
    assign_moved = 0;
    copied_into = 0;
    assign_copied = 0;
    destructed = false;
  };

  reset();

  struct Expected {
    unsigned moved_into = 0;
    unsigned assign_moved = 0;
    unsigned copied_into = 0;
    unsigned assign_copied = 0;
    bool destructed = 0;

    Expected s_moved_into(unsigned in) {
      Expected other = *this;
      other.moved_into = in;
      return other;
    }
    Expected s_assign_moved(unsigned in) {
      Expected other = *this;
      other.assign_moved = in;
      return other;
    }
    Expected s_copied_into(unsigned in) {
      Expected other = *this;
      other.copied_into = in;
      return other;
    }
    Expected s_assign_copied(unsigned in) {
      Expected other = *this;
      other.assign_copied = in;
      return other;
    }
    Expected s_destructed(bool in) {
      Expected other = *this;
      other.destructed = in;
      return other;
    }
  };

  auto check = [&](const Expected &expected, unsigned line_number) {
    std::stringstream info_ss;
    info_ss << "outer line num: " << line_number;
    std::string info = info_ss.str();
    EXPECT_EQ(moved_into, expected.moved_into) << info;
    EXPECT_EQ(assign_moved, expected.assign_moved) << info;
    EXPECT_EQ(copied_into, expected.copied_into) << info;
    EXPECT_EQ(assign_copied, expected.assign_copied) << info;
    EXPECT_EQ(destructed, expected.destructed) << info;

    reset();
  };

  CountCalls<false>(&moved_into, &assign_moved, &copied_into, &assign_copied,
                    &destructed);

  check(Expected{}.s_destructed(true), __LINE__);

  Optional(CountCalls<false>(&moved_into, &assign_moved, &copied_into,
                             &assign_copied, &destructed));

  check(Expected{}.s_moved_into(1).s_destructed(true), __LINE__);

  {
    Optional a(CountCalls<false>(&moved_into, &assign_moved, &copied_into,
                                 &assign_copied, &destructed));
    check(Expected{}.s_moved_into(1).s_destructed(false), __LINE__);
    EXPECT_TRUE(a.has_value());

    a = nullopt_value;

    check(Expected{}.s_destructed(true), __LINE__);
    EXPECT_FALSE(a.has_value());
  }

  unsigned count = 2;
  {
    Optional a(CountCalls<false>(&moved_into, &assign_moved, &copied_into,
                                 &assign_copied, &destructed));

    check(Expected{}.s_moved_into(1), __LINE__);
    EXPECT_TRUE(a.has_value());

    for (unsigned i = 0; i < count; ++i) {
      a = CountCalls<false>(&moved_into, &assign_moved, &copied_into,
                            &assign_copied, &destructed);
      check(Expected{}.s_assign_moved(1).s_moved_into(1), __LINE__);
      EXPECT_TRUE(a.has_value());
    }
  }

  check(Expected{}.s_destructed(true), __LINE__);

  {
    Optional a(CountCalls<false>(&moved_into, &assign_moved, &copied_into,
                                 &assign_copied, &destructed));
    EXPECT_TRUE(a.has_value());

    // avoid warning
    auto assign = [](auto &l, auto &r) { l = std::move(r); };
    assign(a, a);

    check(Expected{}.s_moved_into(true), __LINE__);
    EXPECT_TRUE(a.has_value());
  }

  check(Expected{}.s_destructed(true), __LINE__);

  {
    Optional a(CountCalls<true>(&moved_into, &assign_moved, &copied_into,
                                &assign_copied, &destructed));
    check(Expected{}.s_moved_into(1), __LINE__);
    EXPECT_TRUE(a.has_value());

    {
      const CountCalls<true> other(&moved_into, &assign_moved, &copied_into,
                                   &assign_copied, &destructed);
      {
        Optional b(other);

        check(Expected{}.s_copied_into(1), __LINE__);
        EXPECT_TRUE(b.has_value());

        a = b;

        check(Expected{}.s_assign_copied(1), __LINE__);
        EXPECT_TRUE(a.has_value());
        EXPECT_TRUE(b.has_value());

        a = nullopt_value;

        check(Expected{}.s_destructed(true), __LINE__);
        EXPECT_FALSE(a.has_value());
        EXPECT_TRUE(b.has_value());

        a = b;

        check(Expected{}.s_copied_into(1), __LINE__);
        EXPECT_TRUE(a.has_value());
        EXPECT_TRUE(b.has_value());
      }

      check(Expected{}.s_destructed(true), __LINE__);
    }

    check(Expected{}.s_destructed(true), __LINE__);
  }

  check(Expected{}.s_destructed(true), __LINE__);
  
  {
    Optional a(
        Optional(CountCalls<true>(&moved_into, &assign_moved, &copied_into,
                                  &assign_copied, &destructed)));
    check(Expected{}.s_moved_into(1), __LINE__);
  }

  check(Expected{}.s_destructed(true), __LINE__);
}
