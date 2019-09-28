#include <optional>
#include <ostream>
#include <variant>
#include "helpers.hpp"

struct tennis {
  enum class points { love, fifteen, thirty };
  enum class players { one, two };

  struct normal {
    points one;
    points two;
  };

  struct forty {
    players leader;
    points  trailer;
  };

  struct deuce {};

  struct advantage {
    players leader;
  };

  struct over {
    players winner;
  };

  template <class T>
  constexpr bool is() const noexcept {
    return std::holds_alternative<T>(state_);
  }

  template <class T>
  std::optional<T> get() noexcept {
    if (T* ptr = std::get_if<T>(&state_)) {
      return *ptr;
    } else {
      return {};
    }
  }

  void score(players player) noexcept;

  constexpr bool game_over() const noexcept { return is<over>(); }

  std::optional<players> winner() const noexcept {
    if (game_over()) {
      return std::get<over>(state_).winner;
    } else {
      return {};
    }
  }

  tennis() noexcept
    : state_{normal{points::love, points::love}} {}

private:
  using state_t = std::variant<normal, forty, deuce, advantage, over>;
  state_t state_;
};

void tennis::score(players player) noexcept {
  state_ = variant::match(state_,
    [=](normal const& score) -> state_t {
      switch (player) {
      case players::one:
        switch (score.one) {
        case points::love: return normal{points::fifteen, score.two};
        case points::fifteen: return normal{points::thirty, score.two};
        case points::thirty: return forty{players::one, score.two};
        }
        break;
      case players::two:
        switch (score.two) {
        case points::love: return normal{score.one, points::fifteen};
        case points::fifteen: return normal{score.one, points::thirty};
        case points::thirty: return forty{players::two, score.one};
        }
        break;
      }
      __assume(0);
    },
    [=](forty const& score) -> state_t {
      if (player == score.leader) {
        return over{score.leader};
      } else {
        switch (score.trailer) {
        case points::love: return forty{score.leader, points::fifteen};
        case points::fifteen: return forty{score.leader, points::thirty};
        case points::thirty: return deuce{};
        }
      }
      __assume(0);
    },
    [=](deuce const&) -> state_t {
      return advantage{player};
    },
    [=](advantage const& score) -> state_t {
      if (player == score.leader) {
        return over{score.leader};
      } else {
        return deuce{};
      }
    },
    [=](over const& score) -> state_t {
      return score;
    });
}

std::ostream& operator<<(std::ostream& os, tennis::points points) {
  switch (points) {
  case tennis::points::love: os << "love"; break;
  case tennis::points::fifteen: os << "fifteen"; break;
  case tennis::points::thirty: os << "thirty"; break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, tennis::players players) {
  switch (players) {
  case tennis::players::one: os << "one"; break;
  case tennis::players::two: os << "two"; break;
  }
  return os;
}

#include "gtest/gtest.h"

TEST(tennis, love_all) {
  tennis t;
  EXPECT_TRUE(t.is<tennis::normal>());
  EXPECT_EQ(t.get<tennis::normal>()->one, tennis::points::love);
  EXPECT_EQ(t.get<tennis::normal>()->two, tennis::points::love);
}

TEST(tennis, fifteen_love) {
  tennis t;
  t.score(tennis::players::one);
  EXPECT_EQ(t.get<tennis::normal>()->one, tennis::points::fifteen);
  EXPECT_EQ(t.get<tennis::normal>()->two, tennis::points::love);
}

TEST(tennis, love_fifteen) {
  tennis t;
  t.score(tennis::players::two);
  EXPECT_EQ(t.get<tennis::normal>()->one, tennis::points::love);
  EXPECT_EQ(t.get<tennis::normal>()->two, tennis::points::fifteen);
}

TEST(tennis, fifteen_all) {
  tennis t;
  t.score(tennis::players::one);
  t.score(tennis::players::two);
  EXPECT_EQ(t.get<tennis::normal>()->one, tennis::points::fifteen);
  EXPECT_EQ(t.get<tennis::normal>()->two, tennis::points::fifteen);
}

TEST(tennis, thirty_love) {
  tennis t;
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  EXPECT_EQ(t.get<tennis::normal>()->one, tennis::points::thirty);
  EXPECT_EQ(t.get<tennis::normal>()->two, tennis::points::love);
}

TEST(tennis, thirty_fifteen) {
  tennis t;
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  t.score(tennis::players::two);
  EXPECT_EQ(t.get<tennis::normal>()->one, tennis::points::thirty);
  EXPECT_EQ(t.get<tennis::normal>()->two, tennis::points::fifteen);
}

TEST(tennis, love_thirty) {
  tennis t;
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  EXPECT_EQ(t.get<tennis::normal>()->one, tennis::points::love);
  EXPECT_EQ(t.get<tennis::normal>()->two, tennis::points::thirty);
}

TEST(tennis, fifteen_thirty) {
  tennis t;
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  t.score(tennis::players::one);
  EXPECT_EQ(t.get<tennis::normal>()->one, tennis::points::fifteen);
  EXPECT_EQ(t.get<tennis::normal>()->two, tennis::points::thirty);
}

TEST(tennis, thirty_all) {
  tennis t;
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  EXPECT_EQ(t.get<tennis::normal>()->one, tennis::points::thirty);
  EXPECT_EQ(t.get<tennis::normal>()->two, tennis::points::thirty);
}

TEST(tennis, forty_love) {
  tennis t;
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  EXPECT_EQ(t.get<tennis::forty>()->leader, tennis::players::one);
  EXPECT_EQ(t.get<tennis::forty>()->trailer, tennis::points::love);
}

TEST(tennis, forty_fifteen_1) {
  tennis t;
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  t.score(tennis::players::two);
  t.score(tennis::players::one);
  EXPECT_EQ(t.get<tennis::forty>()->leader, tennis::players::one);
  EXPECT_EQ(t.get<tennis::forty>()->trailer, tennis::points::fifteen);
}

TEST(tennis, forty_thirty_1) {
  tennis t;
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  t.score(tennis::players::one);
  EXPECT_EQ(t.get<tennis::forty>()->leader, tennis::players::one);
  EXPECT_EQ(t.get<tennis::forty>()->trailer, tennis::points::thirty);
}

TEST(tennis, forty_fifteen_2) {
  tennis t;
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  t.score(tennis::players::two);
  EXPECT_EQ(t.get<tennis::forty>()->leader, tennis::players::one);
  EXPECT_EQ(t.get<tennis::forty>()->trailer, tennis::points::fifteen);
}

TEST(tennis, forty_thirty_2) {
  tennis t;
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  EXPECT_EQ(t.get<tennis::forty>()->leader, tennis::players::one);
  EXPECT_EQ(t.get<tennis::forty>()->trailer, tennis::points::thirty);
}

TEST(tennis, love_forty) {
  tennis t;
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  EXPECT_EQ(t.get<tennis::forty>()->leader, tennis::players::two);
  EXPECT_EQ(t.get<tennis::forty>()->trailer, tennis::points::love);
}

TEST(tennis, fifteen_forty_1) {
  tennis t;
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  t.score(tennis::players::one);
  t.score(tennis::players::two);
  EXPECT_EQ(t.get<tennis::forty>()->leader, tennis::players::two);
  EXPECT_EQ(t.get<tennis::forty>()->trailer, tennis::points::fifteen);
}

TEST(tennis, thirty_forty_1) {
  tennis t;
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  t.score(tennis::players::two);
  EXPECT_EQ(t.get<tennis::forty>()->leader, tennis::players::two);
  EXPECT_EQ(t.get<tennis::forty>()->trailer, tennis::points::thirty);
}

TEST(tennis, fifteen_forty_2) {
  tennis t;
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  t.score(tennis::players::one);
  EXPECT_EQ(t.get<tennis::forty>()->leader, tennis::players::two);
  EXPECT_EQ(t.get<tennis::forty>()->trailer, tennis::points::fifteen);
}

TEST(tennis, thirty_forty_2) {
  tennis t;
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  EXPECT_EQ(t.get<tennis::forty>()->leader, tennis::players::two);
  EXPECT_EQ(t.get<tennis::forty>()->trailer, tennis::points::thirty);
}

TEST(tennis, deuce_1) {
  tennis t;
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  t.score(tennis::players::one);
  t.score(tennis::players::two);
  EXPECT_TRUE(t.is<tennis::deuce>());
}

TEST(tennis, deuce_2) {
  tennis t;
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  EXPECT_TRUE(t.is<tennis::deuce>());
}

TEST(tennis, deuce_3) {
  tennis t;
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  EXPECT_TRUE(t.is<tennis::deuce>());
}

TEST(tennis, advantage_1) {
  tennis t;
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  t.score(tennis::players::one);
  EXPECT_EQ(t.get<tennis::advantage>()->leader, tennis::players::one);
}

TEST(tennis, advantage_2) {
  tennis t;
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  EXPECT_EQ(t.get<tennis::advantage>()->leader, tennis::players::two);
}

TEST(tennis, winner_1) {
  tennis t;
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  EXPECT_TRUE(t.game_over());
  EXPECT_EQ(*t.winner(), tennis::players::one);
}

TEST(tennis, winner_2) {
  tennis t;
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  EXPECT_TRUE(t.game_over());
  EXPECT_EQ(*t.winner(), tennis::players::two);
}

