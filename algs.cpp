#include "helpers.hpp"
#include <algorithm>
#include <benchmark/benchmark.h>
#include <execution>
#include <numeric>
#include <random>
#include <string>
#include <vector>

struct data_element_t {
  unsigned long long id;
  std::string        name;

  data_element_t(unsigned long long i = 0)
    : id(i) {
    name = "Element" + std::to_string(i);
  }

  unsigned long long operator()() const noexcept { return id; }
};

auto operator++(data_element_t& v) {
  using namespace std::string_literals;
  ++v.id;
  v.name = "Element"s + std::to_string(v.id);
  return v;
}

template <class T, class BinaryOperation = std::greater<T>>
class comparator {
public:
  comparator(T threshold, BinaryOperation op = std::greater<T>())
    : threshold_(std::move(threshold))
    , op_(std::move(op)) {}

  template <typename U>
  bool operator()(U&& u) const {
    return op_(std::forward<U>(u)(), threshold_);
  }

private:
  T               threshold_;
  BinaryOperation op_;
};

class Fixture : public benchmark::Fixture {
public:
  static constexpr unsigned long long kThreshold = 100000;
  std::vector<data_element_t>         elements;

  void SetUp(benchmark::State const&) override {
    elements.resize(1000000);
    std::iota(elements.begin(), elements.end(), data_element_t{1});
    std::reverse(elements.begin(), elements.end());
  }
};

BENCHMARK_F(Fixture, StdAnyOfEarly)(benchmark::State& state) {
  for (auto _ : state) {
    bool b = std::any_of(elements.begin(), elements.end(),
                         [](auto&& e) { return e.id < kThreshold; });
    if (!b) {
      state.SkipWithError("any_of returned false which shouldn't happen");
      break;
    }
  }
}

BENCHMARK_F(Fixture, MyAnyOfEarly)(benchmark::State& state) {
  for (auto _ : state) {
    bool b = algorithm::any_of(elements.begin(), elements.end(),
                               [](auto&& e) { return e.id < kThreshold; });
    if (!b) {
      state.SkipWithError("any_of returned false which shouldn't happen");
      break;
    }
  }
}

BENCHMARK_F(Fixture, StdAllOfEarly)(benchmark::State& state) {
  for (auto _ : state) {
    bool b = std::all_of(elements.begin(), elements.end(),
                         [](auto&& e) { return e.id < kThreshold; });
    if (b) {
      state.SkipWithError("all_of returned true which shouldn't happen");
      break;
    }
  }
}

BENCHMARK_F(Fixture, StdAllOfGenericEarly)(benchmark::State& state) {
  for (auto _ : state) {
    bool b = std::all_of(elements.begin(), elements.end(),
                         [](auto&& e) { return e() < kThreshold; });
    if (b) {
      state.SkipWithError("all_of returned true which shouldn't happen");
      break;
    }
  }
}

BENCHMARK_F(Fixture, StdAllOfComparatorEarly)(benchmark::State& state) {
  for (auto _ : state) {
    bool b = std::all_of(elements.begin(), elements.end(),
                         comparator(kThreshold, std::less<>()));
    if (b) {
      state.SkipWithError("all_of returned true which shouldn't happen");
      break;
    }
  }
}

BENCHMARK_F(Fixture, MyAllOfEarly)(benchmark::State& state) {
  for (auto _ : state) {
    bool b = algorithm::all_of(elements.begin(), elements.end(),
                               [](auto&& e) { return e.id > kThreshold; });
    if (b) {
      state.SkipWithError("all_of returned true which shouldn't happen");
      break;
    }
  }
}

BENCHMARK_F(Fixture, StdAllOfNoEarly)(benchmark::State& state) {
  for (auto _ : state) {
    bool b = std::all_of(elements.begin(), elements.end(),
                         [](auto&& e) { return e.id > 0; });
    if (!b) {
      state.SkipWithError("all_of returned false which shouldn't happen");
      break;
    }
  }
}

BENCHMARK_F(Fixture, MyAllOfNoEarly)(benchmark::State& state) {
  for (auto _ : state) {
    bool b = algorithm::all_of(elements.begin(), elements.end(),
                               [](auto&& e) { return e.id > 0; });
    if (!b) {
      state.SkipWithError("all_of returned false which shouldn't happen");
      break;
    }
  }
}

BENCHMARK_MAIN();
