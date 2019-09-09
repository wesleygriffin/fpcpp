#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

enum directions { left, right, up, down };
enum types { start, wall, path, leave };

struct maze_t;

struct position_t {
private:
  int x_;
  int y_;

public:
  constexpr position_t(int x, int y) noexcept
    : x_{x}
    , y_{y} {}

  // clang-format off
  constexpr position_t(position_t const& original, directions direction) noexcept
    : x_{direction == directions::left  ? original.x_ - 1 :
         direction == directions::right ? original.x_ + 1 :
                                          original.x_}
    , y_{direction == directions::up    ? original.y_ - 1 :
         direction == directions::down  ? original.y_ + 1 :
                                          original.y_}
  {}
  // clang-format on

  int x() const noexcept { return x_; }
  int y() const noexcept { return y_; }

  auto next(directions direction, maze_t const& maze) const noexcept -> position_t;
};

struct maze_t {
private:
  int                width_;
  int                height_;
  std::vector<types> maze_;

public:
  maze_t(int width, int height, std::initializer_list<types> maze)
    : width_{width}
    , height_{height}
    , maze_{maze} {}

  bool is_leave(position_t const& position) const noexcept {
    return is(types::leave, position);
  }
  bool is_wall(position_t const& position) const noexcept {
    return is(types::wall, position);
  }

  bool is(types type, position_t const& position) const noexcept {
    return maze_[position.y() * size_t(width_) + position.x()] == type;
  }

  position_t start_position() const {
    for (int j = 0; j < height_; ++j) {
      for (int i = 0; i < width_; ++i) {
        auto const position = position_t{i, j};
        if (is(types::start, position)) return position;
      }
    }
    throw std::runtime_error("No start position in maze");
  }

  static directions get_direction() {
    char c = 'a';
    while (c != 'h' && c != 'j' && c != 'k' && c != 'l') {
      std::cout << "Direction [hjkl]? ";
      std::cin >> c;
    }
    std::cout << "\n";

    switch (c) {
    case 'h': return directions::left;
    case 'j': return directions::down;
    case 'k': return directions::up;
    case 'l': return directions::right;
    default: throw std::runtime_error("Invalid direction input");
    }
  }

  void draw(position_t const& position, directions direction) const noexcept {
    for (int j = 0; j < height_; ++j) {
      for (int i = 0; i < width_; ++i) {
        if (position.y() == j && position.x() == i) {
          switch (direction) {
          case directions::down: std::cout << "V"; break;
          case directions::left: std::cout << "<"; break;
          case directions::right: std::cout << ">"; break;
          case directions::up: std::cout << "^"; break;
          }
        } else {
          switch (maze_[j * size_t(width_) + i]) {
          case types::leave: std::cout << " "; break;
          case types::path: std::cout << " "; break;
          case types::start: std::cout << " "; break;
          case types::wall: std::cout << "X"; break;
          }
        }
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  }

  bool process(position_t const& current_position, directions current_direction) const {
    if (is_leave(current_position)) return true;

    draw(current_position, current_direction);
    auto const direction = maze_t::get_direction();

    return process(current_position.next(direction, *this), direction);
  }
};

auto position_t::next(directions direction, maze_t const& maze) const noexcept -> position_t {
  auto const desired = position_t{*this, direction};
  return maze.is_wall(desired) ? *this : desired;
}

int main(int, char**) {
  // clang-format off
  auto const maze  = maze_t{7, 7, {
    types::wall, types::wall,  types::wall, types::wall, types::wall, types::wall, types::wall,
    types::wall, types::path,  types::wall, types::path, types::path, types::path, types::wall,
    types::wall, types::path,  types::wall, types::path, types::wall, types::wall, types::wall,
    types::wall, types::path,  types::path, types::path, types::path, types::path, types::wall,
    types::wall, types::path,  types::wall, types::path, types::wall, types::wall, types::wall,
    types::wall, types::start, types::wall, types::path, types::path, types::path, types::leave,
    types::wall, types::wall,  types::wall, types::wall, types::wall, types::wall, types::wall,
  }};
  // clang-format on

  if (maze.process(maze.start_position(), directions::down)) { std::cout << "Congratulations!\n"; }
}
