#pragma once

#include <compare>
#include <concepts>

struct MockNoRequirements {
  MockNoRequirements() = delete;
  MockNoRequirements(const MockNoRequirements &) = delete;
  MockNoRequirements(MockNoRequirements &&) = delete;
  ~MockNoRequirements() = delete;
};

struct MockMovable {
  MockMovable() = delete;
  MockMovable(const MockMovable &) = delete;
  MockMovable(MockMovable &&) = default;
  MockMovable &operator=(MockMovable &&) = default;
  ~MockMovable() = default;
};

struct MockDefaultInitMovable {
  MockDefaultInitMovable() = default;
  MockDefaultInitMovable(const MockDefaultInitMovable &) = delete;
  MockDefaultInitMovable(MockDefaultInitMovable &&) = default;
  MockDefaultInitMovable &operator=(MockDefaultInitMovable &&) = default;
  ~MockDefaultInitMovable() = default;
};

struct MockCopyable {
  MockCopyable() = delete;
};

struct MockSemiregular {};

struct MockRegular {
  constexpr bool operator==(const MockRegular &) const = default;
};

struct MockTotallyOrdered {
  constexpr auto operator<=>(const MockTotallyOrdered &) const = default;
};
