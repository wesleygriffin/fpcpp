#include <algorithm>
#include <map>
#include <memory>
#include <string>

unsigned int recursive_fib(unsigned int n) noexcept {
  return n == 0 ? 0 :
         n == 1 ? 1 :
                  recursive_fib(n - 1) + recursive_fib(n - 2);
}

class fib_cache {
public:
  fib_cache() noexcept
    : prev_(0)
    , last_(1)
    , size_(2) {}

  size_t size() const noexcept { return size_; }

  unsigned int operator[] (unsigned int n) const {
    // clang-format off
    return n == size_ - 1 ? last_ :
           n == size_ - 2 ? prev_ :
                            0;
    // clang-format on
  }

  void push_back(unsigned int value) noexcept {
    prev_ = last_;
    last_ = value;
    size_++;
  }

  unsigned int back() const noexcept { return last_; }

private:
  unsigned int prev_, last_;
  size_t size_;
};

unsigned int fib(fib_cache& memo, unsigned int n) noexcept {
  if (memo.size() > n) {
    return memo[n];
  } else  {
    memo.push_back(fib(memo, n - 1) + fib(memo, n - 2));
    return memo.back();
  }
}

template <class T>
class matrix {
public:
  matrix(unsigned int rows, unsigned int cols)
    : rows_(rows)
    , cols_(cols)
    , data_(new T[size_t(rows_) * size_t(cols_)]) {
    std::fill_n(data_.get(), rows_ * cols_, std::numeric_limits<T>::max());
  }

  struct index {
    size_t row;
    size_t col;
    constexpr index(unsigned int row, unsigned int col) noexcept
      : row(row)
      , col(col) {}
  };

  T const& operator[](index idx) const noexcept {
    return data_[idx.row * cols_ + idx.col];
  }

  T& operator[](index idx) noexcept {
    return data_[idx.row * cols_ + idx.col];
  }

private:
  unsigned int rows_, cols_;
  std::unique_ptr<T[]> data_;
};

size_t lev_memo_count = 0;
// clang-format off
unsigned int lev(matrix<unsigned int>& memo,
                 unsigned int m, unsigned int n, std::string const& a, std::string const& b) noexcept {
  lev_memo_count += 1;
  if (memo[{m, n}] != std::numeric_limits<unsigned int>::max()) {
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
unsigned int lev(unsigned int m, unsigned int n, std::string const& a, std::string const& b) noexcept {
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
      auto result = f(args...);
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

  matrix<unsigned int> lev_memo(unsigned int(a.length() + 1), unsigned int(b.length() + 1));
  std::cout << "lev(" << a.length() << ", " << b.length() << ", '" << a << "', '" << b
            << "'): " << lev(lev_memo, unsigned int(a.length()), unsigned int(b.length()), a, b)
            << " count: " << lev_memo_count << "\n";

  std::cout << "lev(" << a.length() << ", " << b.length() << ", '" << a << "', '" << b
            << "'): " << lev(unsigned int(a.length()), unsigned int(b.length()), a, b)
            << " count: " << lev_count << "\n";
}