#include <algorithm>
#include <map>
#include <memory>
#include <string>

uint32_t recursive_fib(uint32_t n) noexcept {
  return n == 0 ? 0 : n == 1 ? 1 : recursive_fib(n - 1) + recursive_fib(n - 2);
}

class fib_cache {
public:
  fib_cache() noexcept
    : prev_(0)
    , last_(1)
    , size_(2) {}

  size_t size() const noexcept { return size_; }

  uint32_t operator[](uint32_t n) const {
    // clang-format off
    return n == size_ - 1 ? last_ :
           n == size_ - 2 ? prev_ :
                            0;
    // clang-format on
  }

  void push_back(uint32_t value) noexcept {
    prev_ = last_;
    last_ = value;
    size_++;
  }

  uint32_t back() const noexcept { return last_; }

private:
  uint32_t prev_, last_;
  size_t   size_;
};

uint32_t fib(fib_cache& memo, uint32_t n) noexcept {
  if (memo.size() > n) {
    return memo[n];
  } else {
    memo.push_back(fib(memo, n - 1) + fib(memo, n - 2));
    return memo.back();
  }
}

template <class T>
class matrix {
public:
  matrix(uint32_t rows, uint32_t cols)
    : rows_(rows)
    , cols_(cols)
    , data_(new T[size_t(rows_) * size_t(cols_)]) {
    std::fill_n(data_.get(), rows_ * cols_, std::numeric_limits<T>::max());
  }

  struct index {
    size_t row;
    size_t col;
    constexpr index(uint32_t row, uint32_t col) noexcept
      : row(row)
      , col(col) {}
  };

  T const& operator[](index idx) const noexcept { return data_[idx.row * cols_ + idx.col]; }

  T& operator[](index idx) noexcept { return data_[idx.row * cols_ + idx.col]; }

private:
  uint32_t             rows_, cols_;
  std::unique_ptr<T[]> data_;
};

size_t lev_memo_count = 0;
// clang-format off
uint32_t lev(matrix<uint32_t>& memo,
                 uint32_t m, uint32_t n, std::string const& a, std::string const& b) noexcept {
  lev_memo_count += 1;
  if (memo[{m, n}] != std::numeric_limits<uint32_t>::max()) {
    return memo[{m, n}];
  } else {
    memo[{m, n}] = m == 0 ? n // if a is empty, the distance is the length of b
                 : n == 0 ? m // if b is empty, the distance is the length of a
                 : std::min({
                     lev(memo, m - 1, n, a, b) + 1,                         // add a character
                     lev(memo, m, n - 1, a, b) + 1,                         // remove a character
                     lev(memo, m - 1, n - 1, a, b) + (a[m - 1] != b[n - 1]) // change a character
                 });
    return memo[{m, n}];
  }
}
// clang-format on

size_t lev_count = 0;
// clang-format off
uint32_t lev(uint32_t m, uint32_t n, std::string const& a, std::string const& b) noexcept {
    lev_count += 1;
    return         m == 0 ? n // if a is empty, the distance is the length of b
                 : n == 0 ? m // if b is empty, the distance is the length of a
                 : std::min({
                     lev(m - 1, n, a, b) + 1,                         // add a character
                     lev(m, n - 1, a, b) + 1,                         // remove a character
                     lev(m - 1, n - 1, a, b) + (a[m - 1] != b[n - 1]) // change a character
                 });
}
// clang-format on

template <typename Result, typename... Args>
auto memoize(Result (*f)(Args...)) {
  std::map<std::tuple<Args...>, Result> memo;
  return [f, memo](Args... args) mutable -> Result {
    auto const args_tuple = std::make_tuple(args...);
    if (auto memo_result = memo.find(args_tuple); memo_result == memo.end()) {
      auto result      = f(args...);
      memo[args_tuple] = result;
      return result;
    } else {
      return memo_result->second;
    }
  };
}

#include <iostream>

int main() {
  fib_cache fib_memo;
  std::cout << "fib(6): " << fib(fib_memo, 6) << "\n";

  auto memoized_fib = memoize(recursive_fib);
  std::cout << "fib(6): " << memoized_fib(6) << "\n";

  std::string a = "foofoofoo";
  std::string b = "foofoofoo";

  matrix<uint32_t> lev_memo(uint32_t(a.length() + 1), uint32_t(b.length() + 1));
  std::cout << "lev(" << a.length() << ", " << b.length() << ", '" << a << "', '" << b
            << "'): " << lev(lev_memo, uint32_t(a.length()), uint32_t(b.length()), a, b)
            << " count: " << lev_memo_count << "\n";

  std::cout << "lev(" << a.length() << ", " << b.length() << ", '" << a << "', '" << b
            << "'): " << lev(uint32_t(a.length()), uint32_t(b.length()), a, b)
            << " count: " << lev_count << "\n";
}