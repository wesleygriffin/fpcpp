#include <optional>
#include <ostream>
#include <variant>

struct tennis {
  enum class points { love, fifteen, thirty };
  enum class players { one, two };

  struct normal_scoring {
    points one;
    points two;
  };

  struct forty_scoring {
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

  void score(players player);

  constexpr bool game_over() const noexcept { return is<over>(); }

  std::optional<players> winner() const noexcept {
    if (game_over()) {
      return std::get<over>(state_).winner;
    } else {
      return {};
    }
  }

  tennis() noexcept
    : state_{normal_scoring{points::love, points::love}} {}

private:
  std::variant<normal_scoring, forty_scoring, deuce, advantage, over> state_;
};

void tennis::score(players player) {
  if (normal_scoring* ns = std::get_if<normal_scoring>(&state_)) {
    switch (player) {
    case players::one:
      switch (ns->one) {
      case points::love: ns->one = points::fifteen; break;
      case points::fifteen: ns->one = points::thirty; break;
      case points::thirty: state_ = forty_scoring{players::one, ns->two};
      }
      break;
    case players::two:
      switch (ns->two) {
      case points::love: ns->two = points::fifteen; break;
      case points::fifteen: ns->two = points::thirty; break;
      case points::thirty: state_ = forty_scoring{players::two, ns->one};
      }
      break;
    }
  } else if (forty_scoring* fs = std::get_if<forty_scoring>(&state_)) {
    switch (player) {
    case players::one:
      switch (fs->leader) {
      case players::one: state_ = over{players::one}; break;
      case players::two:
        switch (fs->trailer) {
        case points::love: fs->trailer = points::fifteen; break;
        case points::fifteen: fs->trailer = points::thirty; break;
        case points::thirty: state_ = deuce{}; break;
        }
        break;
      }
      break;
    case players::two:
      switch (fs->leader) {
      case players::one:
        switch (fs->trailer) {
        case points::love: fs->trailer = points::fifteen; break;
        case points::fifteen: fs->trailer = points::thirty; break;
        case points::thirty: state_ = deuce{}; break;
        }
        break;
      case players::two: state_ = over{players::two}; break;
      }
      break;
    }
  } else if (deuce* d = std::get_if<deuce>(&state_)) {
    switch (player) {
    case players::one: state_ = advantage{players::one}; break;
    case players::two: state_ = advantage{players::two}; break;
    }
  } else if (advantage* a = std::get_if<advantage>(&state_)) {
    switch (player) {
    case players::one:
      switch (a->leader) {
      case players::one: state_ = over{players::one}; break;
      case players::two: state_ = deuce{}; break;
      }
      break;
    case players::two:
      switch (a->leader) {
      case players::one: state_ = deuce{}; break;
      case players::two: state_ = over{players::two}; break;
      }
      break;
    }
  } else if (over* o = std::get_if<over>(&state_)) {
    // calling score in this state is bad user input...
  } else {
    throw std::logic_error("invalid state");
  }
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
  EXPECT_TRUE(t.is<tennis::normal_scoring>());
  EXPECT_EQ(t.get<tennis::normal_scoring>()->one, tennis::points::love);
  EXPECT_EQ(t.get<tennis::normal_scoring>()->two, tennis::points::love);
}

TEST(tennis, fifteen_love) {
  tennis t;
  t.score(tennis::players::one);
  EXPECT_EQ(t.get<tennis::normal_scoring>()->one, tennis::points::fifteen);
  EXPECT_EQ(t.get<tennis::normal_scoring>()->two, tennis::points::love);
}

TEST(tennis, love_fifteen) {
  tennis t;
  t.score(tennis::players::two);
  EXPECT_EQ(t.get<tennis::normal_scoring>()->one, tennis::points::love);
  EXPECT_EQ(t.get<tennis::normal_scoring>()->two, tennis::points::fifteen);
}

TEST(tennis, fifteen_all) {
  tennis t;
  t.score(tennis::players::one);
  t.score(tennis::players::two);
  EXPECT_EQ(t.get<tennis::normal_scoring>()->one, tennis::points::fifteen);
  EXPECT_EQ(t.get<tennis::normal_scoring>()->two, tennis::points::fifteen);
}

TEST(tennis, thirty_love) {
  tennis t;
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  EXPECT_EQ(t.get<tennis::normal_scoring>()->one, tennis::points::thirty);
  EXPECT_EQ(t.get<tennis::normal_scoring>()->two, tennis::points::love);
}

TEST(tennis, thirty_fifteen) {
  tennis t;
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  t.score(tennis::players::two);
  EXPECT_EQ(t.get<tennis::normal_scoring>()->one, tennis::points::thirty);
  EXPECT_EQ(t.get<tennis::normal_scoring>()->two, tennis::points::fifteen);
}

TEST(tennis, love_thirty) {
  tennis t;
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  EXPECT_EQ(t.get<tennis::normal_scoring>()->one, tennis::points::love);
  EXPECT_EQ(t.get<tennis::normal_scoring>()->two, tennis::points::thirty);
}

TEST(tennis, fifteen_thirty) {
  tennis t;
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  t.score(tennis::players::one);
  EXPECT_EQ(t.get<tennis::normal_scoring>()->one, tennis::points::fifteen);
  EXPECT_EQ(t.get<tennis::normal_scoring>()->two, tennis::points::thirty);
}

TEST(tennis, thirty_all) {
  tennis t;
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  EXPECT_EQ(t.get<tennis::normal_scoring>()->one, tennis::points::thirty);
  EXPECT_EQ(t.get<tennis::normal_scoring>()->two, tennis::points::thirty);
}

TEST(tennis, forty_love) {
  tennis t;
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  EXPECT_EQ(t.get<tennis::forty_scoring>()->leader, tennis::players::one);
  EXPECT_EQ(t.get<tennis::forty_scoring>()->trailer, tennis::points::love);
}

TEST(tennis, forty_fifteen_1) {
  tennis t;
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  t.score(tennis::players::two);
  t.score(tennis::players::one);
  EXPECT_EQ(t.get<tennis::forty_scoring>()->leader, tennis::players::one);
  EXPECT_EQ(t.get<tennis::forty_scoring>()->trailer, tennis::points::fifteen);
}

TEST(tennis, forty_thirty_1) {
  tennis t;
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  t.score(tennis::players::one);
  EXPECT_EQ(t.get<tennis::forty_scoring>()->leader, tennis::players::one);
  EXPECT_EQ(t.get<tennis::forty_scoring>()->trailer, tennis::points::thirty);
}

TEST(tennis, forty_fifteen_2) {
  tennis t;
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  t.score(tennis::players::two);
  EXPECT_EQ(t.get<tennis::forty_scoring>()->leader, tennis::players::one);
  EXPECT_EQ(t.get<tennis::forty_scoring>()->trailer, tennis::points::fifteen);
}

TEST(tennis, forty_thirty_2) {
  tennis t;
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  EXPECT_EQ(t.get<tennis::forty_scoring>()->leader, tennis::players::one);
  EXPECT_EQ(t.get<tennis::forty_scoring>()->trailer, tennis::points::thirty);
}

TEST(tennis, love_forty) {
  tennis t;
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  EXPECT_EQ(t.get<tennis::forty_scoring>()->leader, tennis::players::two);
  EXPECT_EQ(t.get<tennis::forty_scoring>()->trailer, tennis::points::love);
}

TEST(tennis, fifteen_forty_1) {
  tennis t;
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  t.score(tennis::players::one);
  t.score(tennis::players::two);
  EXPECT_EQ(t.get<tennis::forty_scoring>()->leader, tennis::players::two);
  EXPECT_EQ(t.get<tennis::forty_scoring>()->trailer, tennis::points::fifteen);
}

TEST(tennis, thirty_forty_1) {
  tennis t;
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  t.score(tennis::players::two);
  EXPECT_EQ(t.get<tennis::forty_scoring>()->leader, tennis::players::two);
  EXPECT_EQ(t.get<tennis::forty_scoring>()->trailer, tennis::points::thirty);
}

TEST(tennis, fifteen_forty_2) {
  tennis t;
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  t.score(tennis::players::one);
  EXPECT_EQ(t.get<tennis::forty_scoring>()->leader, tennis::players::two);
  EXPECT_EQ(t.get<tennis::forty_scoring>()->trailer, tennis::points::fifteen);
}

TEST(tennis, thirty_forty_2) {
  tennis t;
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  t.score(tennis::players::two);
  t.score(tennis::players::one);
  t.score(tennis::players::one);
  EXPECT_EQ(t.get<tennis::forty_scoring>()->leader, tennis::players::two);
  EXPECT_EQ(t.get<tennis::forty_scoring>()->trailer, tennis::points::thirty);
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

